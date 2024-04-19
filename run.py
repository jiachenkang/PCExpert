import os
import copy

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_info

from config import ex
from datasets.dataset import ShapeNet, ShapeNetPCD
from modules.pc_expert import PCExpert

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    exp_name = f'{_config["exp_name"]}'
    pl.seed_everything(_config["seed"])

    logger = pl.loggers.WandbLogger(
        name=f'{exp_name}_seed{_config["seed"]}',
        save_dir = _config["log_dir"],
        project="PCD_LEARN5_abltn", 
        )
    rank_zero_info(str(_config))

    if _config["pcd_render"]:
        dataset = ShapeNetPCD(_config)
    else:
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                transforms.ToTensor(), 
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        dataset = ShapeNet(_config, transform)
        
    train_loader = DataLoader(dataset, num_workers=_config['num_workers'], batch_size=_config['per_gpu_batchsize'], shuffle=True, drop_last=True)

    model = PCExpert(_config)

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=80,
        verbose=True,
        monitor="val_acc",
        mode="max",
        save_last=True,
        every_n_epochs=2,
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    if isinstance(_config["num_gpus"], list):
        grad_steps = _config["batch_size"] // (
            _config["per_gpu_batchsize"] * len(_config["num_gpus"]) * _config["num_nodes"]
            )
    else:
        grad_steps = _config["batch_size"] // (
            _config["per_gpu_batchsize"] * _config["num_gpus"] * _config["num_nodes"]
            )

    rank_zero_info("grad_steps: {}".format(grad_steps))

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None


    trainer = pl.Trainer(
        accelerator='gpu',
        devices=_config['num_gpus'],
        num_nodes=_config["num_nodes"],
        strategy="ddp",
        precision=_config["precision"],
        benchmark=True,
        max_epochs=_config["max_epoch"] if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=20,
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
    )

    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        for key in model.trainables:
            if key in name:
                param.requires_grad = True

    for name, param in model.named_parameters():
        rank_zero_info("{}\t{}".format(name, param.requires_grad))

    logger.watch(model,log_graph=False)

    trainer.fit(model, train_loader)

