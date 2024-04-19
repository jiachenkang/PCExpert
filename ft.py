import copy
import re
import random

import numpy as np
import torch
import wandb

from config import ex
from runner_finetune import run_net

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    # set random seeds
    random.seed(_config['seed'])
    np.random.seed(_config['seed'])
    torch.manual_seed(_config['seed'])
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed_all(_config['seed'])

    run_id, step_id = re.split('\W+', _config["ckpt_path"])[2], re.split('\W+', _config["ckpt_path"])[-2]
    wandb.init(
        project="PCD_LEARN5_FT", 
        name=f'{_config["exp_name"]}_seed{_config["seed"]}_{run_id}_{step_id}',
        dir=_config["log_dir"],
        config=_config,
        )
    print(str(_config))

    run_net(_config)
