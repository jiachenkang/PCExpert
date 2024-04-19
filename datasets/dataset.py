import os
import glob
import re
import random

import pickle
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile

from util import IO
import data_utils as d_utils

ImageFile.LOAD_TRUNCATED_IMAGES = True

PCD_DIR = 'data/ShapeNet55-34/shapenet_pc'
IMG_DIR = 'data/ShapeNetRenderingV2'

def load_shapenet_data():
    all_pcd_path_list = glob.glob(os.path.join(PCD_DIR,'*.npy'))
    available_pcd_path_list = [ p for p in all_pcd_path_list 
                         if os.path.isfile(os.path.join(IMG_DIR, 
                                re.split('\W+', p)[-3], 
                                re.split('\W+', p)[-2], 
                                'hard/01.png'))]
    with open('data/available_pcd_path_list', 'wb') as file:
        pickle.dump(available_pcd_path_list, file)


def get_render_imgs(pcd_path):
    imgs_root_dir = os.path.join(IMG_DIR, 
                                re.split('\W+', pcd_path)[-3], 
                                re.split('\W+', pcd_path)[-2])
    img_path_list = glob.glob(os.path.join(imgs_root_dir,'hard', '*.png'))

    return img_path_list

def get_rotate_angle(render_img_path):
    meta_dir = os.path.split(render_img_path)[0]
    with open(os.path.join(meta_dir, 'rendering_metadata.txt')) as f:
        meta_l = f.readlines()
    img_dnx = int(render_img_path[-6:-4])
    rotate_angle = meta_l[img_dnx][1:].split(',')[0]
    
    return float(rotate_angle)
    

def rotate_permute(point_t1, point_t2):
    #angles = np.random.uniform(high=360., size=3)
    #rotate = d_utils.PointcloudRotateByDegree(axis=None, angles=angles)
    angles = np.random.uniform(low=-90., high=90.)
    rotate = d_utils.PointcloudRotateByDegree(axis=np.array([1.0, 0.0, 0.0]), angles=angles)
    point_r1 = rotate(point_t1)
    point_r2 = rotate(point_t2)

    return point_r1, point_r2


class ShapeNet(Dataset):
    def __init__(self, args, transform):

        with open ('data/available_pcd_path_list', 'rb') as file:
            self.data = pickle.load(file)
        #self.data = load_shapenet_data()
        print(f'[DATASET] {len(self.data)} instances were loaded')

        self.transform = transform
        self.npoints_all = 8192
        self.npoints_sample = args['shapenet_npoints_sample']
        self.reg = args['reg'] 
        self.permutation = np.arange(self.npoints_all)


    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc
        
        
    def __getitem__(self, idx):
        pcd_path = self.data[idx]

        render_img_path = random.choice(get_render_imgs(pcd_path))
        rotate_angle = get_rotate_angle(render_img_path)
        render_img = Image.open(render_img_path).convert('RGB')
        render_img = self.transform(render_img) 

        pc_data = IO.get(pcd_path).astype(np.float32)
        pc_data = self.random_sample(pc_data, self.npoints_sample)
        pc_data = self.pc_norm(pc_data)
        pc_data = torch.from_numpy(pc_data).float()

        angles = np.array([0., rotate_angle, 0.])
        rotate = d_utils.PointcloudRotateByDegree(axis=None, angles=angles)
        pc_data_r = rotate(pc_data)
        target = int((rotate_angle % 360) // 30) # 12 classes

        if self.reg:
            #pc_pair = rotate_permute(pc_data, pc_data_r) 
            return (pc_data, pc_data_r), render_img, target
        else:
            return pc_data_r, render_img, target
    

    def __len__(self):
        return len(self.data)


class ShapeNetPCD(ShapeNet):
    def __init__(self, args):
        super().__init__(args, transform=None)

        with open ('data/all_pcd_path_list', 'rb') as file:
            self.data = pickle.load(file)
        print(f'[DATASET] {len(self.data)} instances were loaded')

    
    def get_brightness(self, data):
        z_min = data[:,2].min()
        z_max = data[:,2].max()
        gray_min = 0.2
        gray_max = 0.9
        a = (gray_min - gray_max) / (z_max - z_min)
        b = gray_max - a * z_min
        return a, b
        
        
    def __getitem__(self, idx):
        pcd_path = self.data[idx]
        pcd_8k = IO.get(pcd_path).astype(np.float32)
        
        pcd_8k = self.pc_norm(pcd_8k)
        pcd_2k = self.random_sample(pcd_8k, self.npoints_sample)
        
        pcd_8k = torch.from_numpy(pcd_8k).float()
        pcd_2k = torch.from_numpy(pcd_2k).float()
        
        rotate_angle_x = np.random.uniform(low=-30., high=0.)
        rotate_angle_y = np.random.uniform(low=0., high=360.)
        rotate_2k = d_utils.PointcloudRotateByDegree(
            axis=None, 
            angles=np.array([0, rotate_angle_y, 0])
            )
        rotate_8k = d_utils.PointcloudRotateByDegree(
            axis=None, 
            angles=np.array([rotate_angle_x, rotate_angle_y, 0])
            )
        target = int((rotate_angle_y % 360) // 30) # 12 classes
        
        pcd_2k_r = rotate_2k(pcd_2k)
        pcd_8k = rotate_8k(pcd_8k)
        a, b = self.get_brightness(pcd_8k)
        rgb = (pcd_8k[:,2] * a + b).unsqueeze(1).repeat(1,3)

        if self.reg:
            return (pcd_2k, pcd_2k_r), (pcd_8k, rgb), target
        else:
            return pcd_2k_r, (pcd_8k, rgb), target
        


def load_modelnet_data(partition):
    BASE_DIR = ''
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', f'ply_data_{partition}*.h5')):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def load_modelnet_data(partition):
    BASE_DIR = ''
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', f'ply_data_{partition}*.h5')):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


class ModelNet40SVM(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_modelnet_data(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


def load_ScanObjectNN(partition):
    BASE_DIR = 'data/ScanObjectNN'
    DATA_DIR = os.path.join(BASE_DIR, 'main_split')
    h5_name = os.path.join(DATA_DIR, f'{partition}.h5')
    f = h5py.File(h5_name)
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    
    return data, label

    
class ScanObjectNNSVM(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_ScanObjectNN(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]