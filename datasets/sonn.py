import numpy as np
import os, sys, h5py
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch


class ScanObjectNN(Dataset):
    def __init__(self, split, transform, _config, **kwargs):
        super().__init__()
        self.subset = split
        self.root = _config['data_root']
        self.transform = transform
        
        if self.subset == 'train':
            h5 = h5py.File(os.path.join(self.root, 'training_objectdataset.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        elif self.subset == 'test':
            h5 = h5py.File(os.path.join(self.root, 'test_objectdataset.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            raise NotImplementedError()

        print(f'Successfully load ScanObjectNN shape of {self.points.shape}')

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])   # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        
        current_points = self.points[idx, pt_idxs].copy()

        current_points = torch.from_numpy(current_points).float()
        if self.transform is not None:
            current_points = self.transform(current_points)
        label = self.labels[idx]
        
        return 'ScanObjectNN', 'sample', (current_points, label)

    def __len__(self):
        return self.points.shape[0]



class ScanObjectNN_hardest(Dataset):
    def __init__(self, split, transform, _config, **kwargs):
        super().__init__()
        self.subset = split
        self.root = _config['data_root']
        self.transform = transform
        
        if self.subset == 'train':
            h5 = h5py.File(os.path.join(self.root, 'training_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        elif self.subset == 'test':
            h5 = h5py.File(os.path.join(self.root, 'test_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            raise NotImplementedError()

        print(f'Successfully load ScanObjectNN_hardest shape of {self.points.shape}')

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])   # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        
        current_points = self.points[idx, pt_idxs].copy()
        
        current_points = torch.from_numpy(current_points).float()
        if self.transform is not None:
            current_points = self.transform(current_points)
        label = self.labels[idx]
        
        return 'ScanObjectNN', 'sample', (current_points, label)

    def __len__(self):
        return self.points.shape[0]

