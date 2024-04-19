import os

import h5py
import numpy as np
import torch
import torch.optim as optim
from timm.scheduler import CosineLRScheduler


class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)
       
    # References: https://github.com/dimatura/pypcd/blob/master/pypcd/pypcd.py#L275
    # Support PCD files without compression ONLY!
    # @classmethod
    # def _read_pcd(cls, file_path):
    #     pc = open3d.io.read_point_cloud(file_path)
    #     ptcloud = np.array(pc.points)
    #     return ptcloud

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]



class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


class AverageMeter(object):
    def __init__(self, items=None):
        self.items = items
        self.n_items = 1 if items is None else len(items)
        self.reset()

    def reset(self):
        self._val = [0] * self.n_items
        self._sum = [0] * self.n_items
        self._count = [0] * self.n_items

    def update(self, values):
        if type(values).__name__ == 'list':
            for idx, v in enumerate(values):
                self._val[idx] = v
                self._sum[idx] += v
                self._count[idx] += 1
        else:
            self._val[0] = values
            self._sum[0] += values
            self._count[0] += 1

    def val(self, idx=None):
        if idx is None:
            return self._val[0] if self.items is None else [self._val[i] for i in range(self.n_items)]
        else:
            return self._val[idx]

    def count(self, idx=None):
        if idx is None:
            return self._count[0] if self.items is None else [self._count[i] for i in range(self.n_items)]
        else:
            return self._count[idx]

    def avg(self, idx=None):
        if idx is None:
            return self._sum[0] / self._count[0] if self.items is None else [
                self._sum[i] / self._count[i] for i in range(self.n_items)
            ]
        else:
            return self._sum[idx] / self._count[idx]


class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda()) + torch.from_numpy(xyz2).float().cuda()
            
        return pc


class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)



def build_optimizer(base_model, args):
    def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
        decay = []
        no_decay = []
        for name, param in model.module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                # print(name)
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]
    param_groups = add_weight_decay(base_model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    return optimizer


def build_opti_sche(models, config):

    decay = []
    no_decay = []
    def add_weight_decay(model, skip_list=()):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or 'embedding' in name or name in skip_list:
                # print(name)
                no_decay.append(param)
            else:
                decay.append(param)
    
    lr = config['learning_rate']
    wd = config['weight_decay']

    optim_type = config["optim_type"]
    for m in models:
        add_weight_decay(m)
    param_groups = [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': wd}]


    if optim_type == "adamw":
        optimizer = optim.AdamW(param_groups, lr=lr, eps=1e-8, betas=(0.9, 0.98))
    elif optim_type == "adam":
        optimizer = optim.Adam(param_groups, lr=lr)
    elif optim_type == "sgd":
        optimizer = optim.SGD(param_groups, lr=lr, momentum=0.9)

    max_epochs = config["max_epoch"]
    warmup_epochs = config["warmup_epochs"]
    print("warmup_epochs:{} \t max_epochs:{}".format(warmup_epochs, max_epochs))

    scheduler = CosineLRScheduler(
            optimizer,
            t_initial=max_epochs,
            cycle_mul=1.,
            lr_min=1e-6,
            # lr_min=1e-7,
            cycle_decay=0.1,
            warmup_lr_init=1e-6,
            # warmup_lr_init=1e-7,
            warmup_t=warmup_epochs,
            cycle_limit=1,
            t_in_epochs=True,
            )
    
    return optimizer, scheduler
