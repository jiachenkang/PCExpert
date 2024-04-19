import re
import os

import numpy as np
import time
import datetime
import torch
import torch.nn as nn
import sklearn.metrics as metrics
import wandb

from datasets.finetune_dataloader import get_dataloader
from util import build_opti_sche, AverageMeter
from modules.pc_expert import PCExpert

class Acc_Metric:
    def __init__(self, acc=0., acc_avg=0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
            self.acc_avg = acc['acc_avg']
        elif type(acc).__name__ == 'Acc_Metric':
            self.acc = acc.acc
            self.acc_avg = acc.acc_avg
        else:
            self.acc = acc
            self.acc_avg = acc_avg

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        _dict['acc_avg'] = self.acc_avg
        return _dict


def run_net(_config):
    device = torch.device(type='cuda', index=_config['gpu_id'])
    train_dataloader, test_dataloader = get_dataloader(_config)

    #load pretrained model weights
    pcexpert = PCExpert.load_from_checkpoint(_config['ckpt_path'], strict=False)
    transformer = pcexpert.transformer
    trainables = [p.replace('transformer.', '') for p in pcexpert.trainables]

    for param in transformer.parameters():
        param.requires_grad = False

    if _config['tuning_level'] == 'full':
        for name, param in transformer.named_parameters():
            for key in trainables:
                if key in name:
                    param.requires_grad = True

    for name, param in transformer.named_parameters():
        if param.requires_grad == True:
            print(name)


    # mlp model
    if _config['tuning_level'] == 'linear':
        base_model = nn.Linear(1024, _config['finetune_cls_dim'])
    else:
        # mlp-3 head (original head)
        base_model = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, _config['finetune_cls_dim'])
        )
    wandb.watch(base_model)

    loss_fn = nn.CrossEntropyLoss()
    
    # parameter setting
    start_epoch = 0
    best_epoch = 0
    best_metrics = Acc_Metric(0., 0.)
    best_metrics_vote = Acc_Metric(0., 0.)
    metrics = Acc_Metric(0., 0.)

    time_sec_tot = 0.

    transformer.to(device)
    base_model.to(device)
    loss_fn.to(device)

    # optimizer & scheduler
    optimizer, scheduler = build_opti_sche((transformer,base_model), _config)

    step_per_update = _config["batch_size"] // _config["per_gpu_batchsize"]
    print("grad_steps: {}".format(step_per_update))

    transformer.zero_grad()
    base_model.zero_grad()

    for epoch in range(start_epoch, _config['max_epoch'] + 1):
        if _config['tuning_level'] == 'full':
            transformer.train()
        else: 
            transformer.eval()
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        n_batches = len(train_dataloader)

        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            data_time.update(time.time() - batch_start_time)

            points = data[0].to(device)
            label = data[1].to(device)
            
            _, pcd_features = transformer(points, modality_type='pcd')
            ret = base_model(pcd_features)
            loss = loss_fn(ret, label.long())
            pred = ret.argmax(-1)
            acc = (pred == label).sum() / float(label.size(0))

            _loss = loss
            _loss.backward()
            
            if num_iter == step_per_update:
                num_iter = 0
                optimizer.step()
                transformer.zero_grad()
                base_model.zero_grad()

                wandb_log_batch = {}
                wandb_log_batch['train_loss_step'] = loss.item()
                wandb_log_batch['train_acc_step'] = acc.item()
                wandb_log_batch['lr'] = optimizer.param_groups[0]['lr']
                wandb.log(wandb_log_batch)

            losses.update([loss.item(), acc.item()])

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

        scheduler.step(epoch)
        epoch_end_time = time.time()

        wandb_log = {}
        wandb_log['epoch'] = epoch
        wandb_log['train_loss_epoch'] = losses.avg(0)
        wandb_log['train_acc_epoch'] = losses.avg(1)

        # recording time
        epoch_time = epoch_end_time - epoch_start_time
        time_sec_tot += epoch_time
        time_sec_avg = time_sec_tot / (epoch - start_epoch + 1)
        eta_sec = time_sec_avg * (_config['max_epoch'] - epoch)
        eta_str = str(datetime.timedelta(seconds=int(eta_sec)))

        print('[Training] EPOCH: %d ETA: %s EpochTime: %.3f (s) [Loss,Acc] = %s' %
            (epoch, eta_str, epoch_time, ['%.4f' % l for l in losses.avg()]))
        
        metrics = validate(transformer, base_model, test_dataloader, device, epoch)

        wandb_log['valid_acc'] = metrics.acc
        wandb.log(wandb_log)

        better = metrics.better_than(best_metrics)
        # Save ckeckpoints
        if better:
            best_metrics = metrics
            best_epoch = epoch
            save_checkpoint(_config["ckpt_path"], _config["log_dir"], _config["exp_name"], _config["fold"], base_model, epoch, metrics, best_metrics, )

    print("[Training] Best OA=%.4f  mAcc=%.4f" % (best_metrics.acc, best_metrics.acc_avg)) 



def validate(transformer, base_model, test_dataloader, device, epoch):
    transformer.eval()
    base_model.eval()
    
    test_pred  = []
    test_label = []

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].to(device)
            label = data[1].to(device)

            _, pcd_features = transformer(points, modality_type='pcd')
            logits = base_model(pcd_features)
            target = label.view(-1)

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())
        
        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)
        test_label, test_pred = test_label.cpu().numpy(), test_pred.cpu().numpy()

        acc = metrics.accuracy_score(test_label, test_pred)
        acc_avg = metrics.balanced_accuracy_score(test_label, test_pred)

        print('[Validation] EPOCH: %d  OA=%.4f  mAcc=%.4f' % (epoch, acc, acc_avg))

    return Acc_Metric(acc, acc_avg)


def save_checkpoint(ckpt_path, log_dir, exp_name, fold, base_model, epoch, metrics, best_metrics, ):
    run_id, step_id = re.split('\W+', ckpt_path)[2], re.split('\W+', ckpt_path)[-2]
    save_ckpt_path = os.path.join(log_dir, 'PCD_LEARN5', run_id, exp_name)
    os.makedirs(save_ckpt_path, exist_ok=True)
    file_name = f'step{step_id}_best.pth' if fold == -1 else f'step{step_id}_fold{fold}_best.pth'
    
    torch.save({
        'state_dict' : base_model.state_dict(),
        'epoch' : epoch,
        'metrics' : metrics.state_dict() if metrics is not None else dict(),
        'best_metrics' : best_metrics.state_dict() if best_metrics is not None else dict(),
                }, os.path.join(save_ckpt_path, file_name))
    print(f"Save checkpoint at {os.path.join(save_ckpt_path, file_name)}")

