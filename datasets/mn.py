'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle

import torch
from torch.utils.data import Dataset


warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc



def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNet(Dataset):
    def __init__(self, split, transform, _config):
        self.root = _config['data_root']
        self.npoints = 1024
        self.num_category = _config['finetune_cls_dim']
        self.subset = split
        self.save_path = os.path.join(self.root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, 8192))
        self.transform = transform

        with open(self.save_path, 'rb') as f:
            self.list_of_points, self.list_of_labels = pickle.load(f)

        print(f'Successfully load ModelNet40 shape of {len(self.list_of_labels)}')


    def __len__(self):
        return len(self.list_of_labels)


    def __getitem__(self, index):
        point_set, label = self.list_of_points[index][:, 0:3], self.list_of_labels[index][0]

        pt_idxs = np.arange(0, point_set.shape[0])  # 8192
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        current_points = point_set[pt_idxs[:self.npoints]].copy() # 1024
        
        current_points = pc_normalize(current_points)
        current_points = torch.from_numpy(current_points).float()
        #if self.transform is not None:
        #    current_points = self.transform(current_points)

        return 'ModelNet', 'sample', (current_points, label)



class ModelNetFewShot(Dataset):
    def __init__(self, split, transform, _config):
        self.root = _config['data_root']
        self.npoints = 1024
        self.num_category = _config['finetune_cls_dim']
        self.subset = split

        self.way = _config['way']
        self.shot = _config['shot']
        self.fold = _config['fold']
        if self.way == -1 or self.shot == -1 or self.fold == -1:
            raise RuntimeError()

        self.pickle_path = os.path.join(self.root, f'{self.way}way_{self.shot}shot', f'{self.fold}.pkl')
        print('Load processed data from %s...' % self.pickle_path)

        with open(self.pickle_path, 'rb') as f:
            self.dataset = pickle.load(f)[self.subset]

        print(f'Successfully load ModelNetFewShot {split} set shape of {len(self.dataset)}.')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        points, label, _ = self.dataset[index]
        points = points[:, 0:3]

        pt_idxs = np.arange(0, points.shape[0])   # 8192
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs[:self.npoints]].copy() # 1024

        current_points = pc_normalize(current_points)
        current_points = torch.from_numpy(current_points).float()

        return 'ModelNetFewShot', 'sample', (current_points, label)