import copy
from collections import OrderedDict
import numpy as np
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.distributed as dist

import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_info
import torchmetrics

from timm.models.layers import trunc_normal_

#from transformers import get_cosine_schedule_with_warmup
from timm.scheduler import CosineLRScheduler

from modules.pc_expert import PCExpert



class PCExpertFT(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        self.pcexpert = PCExpert.load_from_checkpoint(args['ckpt_path'], strict=False)

        self.loss_fn = nn.CrossEntropyLoss()
        self.cls_dim = args['finetune_cls_dim']
        self.trn_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.cls_dim)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.cls_dim)


        if args['tuning_level'] == 'linear':
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(1024, self.cls_dim)
            )
        else:
            # mlp-3 head (original head)
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )

        for m in self.cls_head_finetune:
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)


        if args['tuning_level'] == 'full':
            self.trainables = ['pcexpert.' + k for k in self.pcexpert.trainables]
            self.trainables.extend(['cls_head_finetune'])
        else:
            self.trainables = ['cls_head_finetune']


        self.lr = args["learning_rate"]
        self.decay = []
        self.no_decay = []
        self._add_weight_decay(self)
        

    def training_step(self, batch, batch_idx):
        _, _, (pcd, target) = batch
        _, pcd_features = self.pcexpert.transformer(pcd, modality_type='pcd')
        preds = self.cls_head_finetune(pcd_features)

        loss = self.loss_fn(preds, target.long())
        self.trn_acc(preds, target)

        self.log("loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_acc', self.trn_acc, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        _, _, (pcd, target) = batch
        _, pcd_features = self.pcexpert.transformer(pcd, modality_type='pcd')
        preds = self.cls_head_finetune(pcd_features)
        
        self.val_acc(preds, target)

        self.log('valid_acc', self.val_acc, on_epoch=True)


    def on_save_checkpoint(self, checkpoint):
        pretrained_keys = [k for k in checkpoint['state_dict'] if k not in self.trainables]
        for key in pretrained_keys:
            del checkpoint['state_dict'][key]

    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)


    def _add_weight_decay(self, model):
        no_decay_list = [
            "bias",
            "ln_1.weight",
            "ln_2.weight",
            'bn_',
            "ln_pre.weight",
            "ln_post.weight",
            "embedding",
        ]
        for name, param in model.named_parameters():
            #if not any(t in name for t in self.trainables):
                #continue # frozen weight
            if len(param.shape) == 1 or any(nd in name for nd in no_decay_list):
                self.no_decay.append(param)
            else:
                self.decay.append(param)

    def configure_optimizers(self):
        lr = self.lr
        wd = self.hparams.args["weight_decay"]

        optim_type = self.hparams.args["optim_type"]

        optimizer_grouped_parameters = [
            {
                "params": self.decay,
                "weight_decay": wd,
                "lr": lr,
            },
            {
                "params": self.no_decay,
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]

        if optim_type == "adamw":
            optimizer = AdamW(
                optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
            )
        elif optim_type == "adam":
            optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
        elif optim_type == "sgd":
            optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

        max_epochs = self.trainer.max_epochs
        warmup_epochs = self.hparams.args["warmup_epochs"]
        rank_zero_info("warmup_epochs:{} \t max_epochs:{}".format(warmup_epochs, max_epochs))

        scheduler = CosineLRScheduler(
                optimizer,
                t_initial=max_epochs,
                cycle_mul=1.,
                # lr_min=1e-6,
                lr_min=1e-7,
                cycle_decay=0.1,
                warmup_lr_init=1e-6,
                # warmup_lr_init=1e-7,
                warmup_t=warmup_epochs,
                cycle_limit=1,
                t_in_epochs=True,
                )

        sched = {"scheduler": scheduler, "interval": "epoch"}

        return (
            [optimizer],
            [sched],
        )
    

    