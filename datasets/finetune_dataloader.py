import torch

from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.data_utils import PointcloudRotate
from . import _datasets

train_transforms = PointcloudRotate()

def get_dataloader(_config):
    dl_key = _config["dataset"]
    assert len(dl_key) > 0

    dataset_cls = _datasets[dl_key]

    train_dataset = dataset_cls(
            split="train",
            transform=train_transforms,
            _config=_config,
        )
    
    val_dataset = dataset_cls(
            split="test",
            transform=None,
            _config=_config,
        )
    
    train_loader = DataLoader(
            train_dataset,
            batch_size=_config["per_gpu_batchsize"],
            shuffle=True,
            num_workers=_config["num_workers"],
            pin_memory=True,
        )
    
    val_loader = DataLoader(
            val_dataset,
            batch_size=_config["per_gpu_batchsize"],
            shuffle=False,
            num_workers=_config["num_workers"],
            pin_memory=True,
        )
    
    return train_loader, val_loader