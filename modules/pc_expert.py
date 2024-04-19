import numpy as np
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.distributed as dist
import torchvision.transforms as transforms

import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_info

from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    NormWeightedCompositor
)

from .multiway_transformer import MultiWayTransformer
from datasets.dataset import ModelNet40SVM, ScanObjectNNSVM


def compute_ctr_loss(feature_a, feature_b, logit_scale):
    logits_per_a = logit_scale * feature_a @ feature_b.t()
    logits_per_b = logits_per_a.t()

    ground_truth = torch.arange(len(logits_per_a)).long().to(device=logits_per_a.get_device())

    loss = (
            F.cross_entropy(logits_per_a.float(), ground_truth)
            + F.cross_entropy(logits_per_b.float(), ground_truth)
        ) / 2
    return loss

def compute_reg_loss(feature_a, feature_b, targets, reg_head):
    distance = F.normalize((feature_a - feature_b), dim=-1)
    outputs = F.normalize(distance @ reg_head, dim=-1)
    loss = (1 - outputs[range(outputs.shape[0]), targets]).mean()
    return loss

class PCExpert(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        self.img_size = args["image_size"]
        if args['pcd_render']:
            self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            self.raster_settings = PointsRasterizationSettings(
                image_size=self.img_size, 
                radius=0.03,
                points_per_pixel = 8
            )
            self.R, self.T = look_at_view_transform(2.2,0,180)

        self.transformer = MultiWayTransformer(
            input_resolution=self.img_size, 
            patch_size=args['patch_size'], 
            num_group=args['num_group'], 
            group_size=args['group_size'], 
            expert_dim=args['expert_dim'],
            img_adapter_att_dim=args['img_adapter_att_dim'],
            img_adapter_mlp_dim=args['img_adapter_mlp_dim'],
            pcd_adapter_att_dim=args['pcd_adapter_att_dim'],
            pcd_adapter_mlp_dim=args['pcd_adapter_mlp_dim'],
            drop_path_rate=args["drop_path_rate"],
            output_dim=args['output_dim'],
            )

        self.pcd_reg_head = nn.Parameter(self.transformer.scale * torch.randn(args['output_dim'], args['num_classes']))

        self.num_layers = self.transformer.layers

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.val_d = {
            'feats_train' : [],
            'labels_train' : [],
            'feats_test' : [],
            'labels_test' : [],
        }

        self.lr = self.hparams.args["learning_rate"]

        self.trainables = self.load_pretrained_weight()
        self.trainables.extend(['pcd_reg_head', 'logit_scale'])


    def load_pretrained_weight(self):
        model_path = self.hparams.args["load_path"]
        with open(model_path, 'rb') as opened_file:
            model = torch.jit.load(opened_file, map_location='cpu').eval()
        state_dict = model.state_dict()
        rank_zero_info(f"Load ckpt from: {model_path}.")
        
        state_dict = self.convert_ckpt(state_dict)
        
        missing_keys, unexpected_keys = self.transformer.load_state_dict(state_dict, strict=False)
        rank_zero_info("missing_keys: {}".format(missing_keys))
        rank_zero_info("unexpected_keys: {}".format(unexpected_keys)) 

        missing_keys = ['transformer.' + k for k in missing_keys]
        return missing_keys

        
    def convert_ckpt(self, state_dict):
        keys_to_remove = [k for k in state_dict if not k.startswith("visual.")]
        for key in keys_to_remove:
            del state_dict[key]
        del state_dict['visual.proj']
        new_state_dict = {}

        for key in state_dict:
            value = state_dict[key]
            if key.startswith('visual.transformer'):
                if 'mlp' in key:
                    new_key = key.replace('visual.transformer.', '').replace('mlp', 'img_mlp')
                    new_state_dict[new_key] = value
                elif 'ln_2' in key:
                    new_key = key.replace('visual.transformer.', '').replace('ln_2', 'img_ln_2')
                    new_state_dict[new_key] = value
                else:
                    new_key = key.replace('visual.transformer.', '')
                    new_state_dict[new_key] = value
            
            elif 'embedding' in key or 'conv1' in key:
                new_key = key.replace('visual', 'img_patch_emb')
                new_state_dict[new_key] = value

            elif 'proj' in key or 'ln_p' in key:
                new_key = key.replace('visual.', 'img_')
                new_state_dict[new_key] = value

        return new_state_dict
    

    def pcd_to_img(self, pcd):
        pcd_8k, rgb_8k = pcd
        pc = Pointclouds(points=pcd_8k, features=rgb_8k)
        
        cameras = FoVPerspectiveCameras(device=pcd_8k.device, R=self.R, T=self.T, znear=1)
        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings),
            compositor=NormWeightedCompositor(background_color=(0.,0.,0.))
        )
        with torch.no_grad():
            imgs = renderer(pc)
        imgs = self.normalize(imgs.permute(0, 3, 1, 2))
        return imgs
    

    def training_step(self, batch, batch_idx):

        (pcd1, pcd2), img, targets = batch
        if self.hparams.args["pcd_render"]:
            img = self.pcd_to_img(img)
        batch_size = pcd1.size()[0]

        pcd = torch.cat((pcd1, pcd2))
        pcd_features, _ = self.transformer(pcd, modality_type='pcd')
        img_features = self.transformer(img, modality_type='img')
        
        pcd_features = pcd_features / pcd_features.norm(dim=-1, keepdim=True)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)

        pcd1_features = pcd_features[:batch_size, :]
        pcd2_features = pcd_features[batch_size: , :]

        loss_reg = compute_reg_loss(pcd1_features, pcd2_features, targets, self.pcd_reg_head)

        logit_scale = self.logit_scale.exp().mean()
        
        if self.hparams.args['aggregate']:
            world_size = dist.get_world_size()
            rank = dist.get_rank()

            # We gather tensors from all gpus to get more negatives to contrast with.
            gathered_img_features = [
                torch.zeros_like(img_features) for _ in range(world_size)
            ]
            gathered_pcd1_features = [
                torch.zeros_like(pcd1_features) for _ in range(world_size)
            ]
            gathered_pcd2_features = [
                torch.zeros_like(pcd2_features) for _ in range(world_size)
            ]
            dist.all_gather(gathered_img_features, img_features)
            dist.all_gather(gathered_pcd1_features, pcd1_features)
            dist.all_gather(gathered_pcd2_features, pcd2_features)

            all_img_features = torch.cat(
                [img_features]
                + gathered_img_features[:rank]
                + gathered_img_features[rank + 1 :]
            )
            all_pcd1_features = torch.cat(
                [pcd1_features]
                + gathered_pcd1_features[:rank]
                + gathered_pcd1_features[rank + 1 :]
            )
            all_pcd2_features = torch.cat(
                [pcd2_features]
                + gathered_pcd2_features[:rank]
                + gathered_pcd2_features[rank + 1 :]
            )

            all_pcd_features = torch.stack([all_pcd1_features,all_pcd2_features]).mean(dim=0)

            # this is needed to send gradients back everywhere.
            loss_cm = compute_ctr_loss(all_img_features, all_pcd_features, logit_scale)
            loss_im = compute_ctr_loss(all_pcd1_features, all_pcd2_features, logit_scale)

        else:
            #pcd_feats = torch.stack([pcd1_features,pcd2_features]).mean(dim=0)
            loss_cm = compute_ctr_loss(img_features, pcd2_features, logit_scale)
            loss_im = compute_ctr_loss(pcd1_features, pcd2_features, logit_scale)

        total_loss = loss_cm + loss_im * self.hparams.args["lamda_im"] + loss_reg * self.hparams.args["lamda_reg"]
        
        self.log("train_loss", total_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("loss_reg", loss_reg, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("loss_cm", loss_cm, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("loss_im", loss_im, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return total_loss
    

    def val_dataloader(self):
        train_val_loader = DataLoader(ModelNet40SVM(partition='train', num_points=1024), num_workers=self.hparams.args['num_workers'], batch_size=120)
        test_val_loader = DataLoader(ModelNet40SVM(partition='test', num_points=1024), num_workers=self.hparams.args['num_workers'], batch_size=120)
        return [train_val_loader, test_val_loader]
    

    def validation_step(self, batch, batch_idx, dataloader_idx):
        data, label = batch
        _, pcd_feature = self.transformer(data, modality_type='pcd')

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # Gather embedings from all gpus
        gathered_features = [
            torch.zeros_like(pcd_feature) for _ in range(world_size)
        ] #(B,D) * gpus
        gathered_labels = [
            torch.zeros_like(label) for _ in range(world_size)
        ] #(B,1) * gpus
        dist.all_gather(gathered_features, pcd_feature)
        dist.all_gather(gathered_labels, label)
        all_features = torch.cat(
            [pcd_feature]
            + gathered_features[:rank]
            + gathered_features[rank + 1 :]
        ) #(B*gpus, D)
        all_labels = torch.cat(
            [label]
            + gathered_labels[:rank]
            + gathered_labels[rank + 1 :]
        ) #(B*gpus,1)

        all_features = all_features.cpu().numpy()
        all_labels_l = list(map(lambda x: x[0],all_labels.cpu().numpy().tolist())) # ModelNet40SVM

        if dataloader_idx == 0:
            for f in all_features:
                self.val_d['feats_train'].append(f)
            self.val_d['labels_train'] += all_labels_l
        else:
            for f in all_features:
                self.val_d['feats_test'].append(f)
            self.val_d['labels_test'] += all_labels_l

    def on_validation_epoch_end(self):
        feats_train = np.array(self.val_d['feats_train'])
        labels_train = np.array(self.val_d['labels_train'])
        feats_test = np.array(self.val_d['feats_test'])
        labels_test = np.array(self.val_d['labels_test'])

        model_tl = SVC(C = 0.1, kernel ='linear')
        model_tl.fit(feats_train, labels_train)
        test_accuracy = model_tl.score(feats_test, labels_test)
        self.log("val_acc", test_accuracy, sync_dist=True)

        self.val_d = {
            'feats_train' : [],
            'labels_train' : [],
            'feats_test' : [],
            'labels_test' : [],
        }

    def on_save_checkpoint(self, checkpoint):
        pretrained_keys = [k for k in checkpoint['state_dict'] if k not in self.trainables]
        for key in pretrained_keys:
            del checkpoint['state_dict'][key]


    def configure_optimizers(self):
        lr = self.lr
        wd = self.hparams.args["weight_decay"]

        no_decay = [
            "bias",
            "ln_1.weight",
            "ln_2.weight",
            "ln_pre.weight",
            "ln_post.weight",
        ]

        end_lr = self.hparams.args["end_lr"]
        decay_power = self.hparams.args["decay_power"]
        optim_type = self.hparams.args["optim_type"]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.transformer.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": wd,
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in self.transformer.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": wd * 0.2, 
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

        if self.trainer.max_steps is None or self.trainer.max_steps==-1:
            max_steps = (
                len(self.trainer.datamodule.train_dataloader())
                * self.trainer.max_epochs
                // self.trainer.accumulate_grad_batches
            )
        else:
            max_steps = self.trainer.max_steps

        warmup_steps = self.hparams.args["warmup_steps"]
        if isinstance(self.hparams.args["warmup_steps"], float):
            warmup_steps = int(max_steps * warmup_steps)
        rank_zero_info("Warmup_steps:{} \t Max_steps:{}".format(warmup_steps, max_steps))

        if decay_power == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=max_steps,
            )
        else:
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=max_steps,
                lr_end=end_lr,
                power=decay_power,
            )

        sched = {"scheduler": scheduler, "interval": "step"}

        return (
            [optimizer],
            [sched],
        )
    


