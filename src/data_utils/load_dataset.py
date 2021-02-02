# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/data_utils/load_dataset.py

import itertools
import os
import h5py as h5
import numpy as np
import random
from scipy import io
from PIL import ImageOps, Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, STL10
from torchvision.datasets import ImageFolder


class FairFaceDataset(Dataset):

    def __init__(self, image_root, exp_type, train = True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.exp_type = exp_type

        if train:
            self.train = True
            self.label_f = os.path.join(image_root, 'fairface_label_train.csv' )
            self.image_dir = os.path.join(image_root, 'train')
        else:
            self.train = False
            self.label_f = os.path.join(image_root, 'fairface_label_val.csv' )
            self.image_dir = os.path.join(image_root, 'val')

        self.parse_label_file()
        self.combine_label_space()

        self.data_list = []
        self.transform = transform

        for image_f in os.listdir(self.image_dir):

            image_fp = os.path.join(self.image_dir, image_f)
            image_key = os.path.join('train' if self.train else 'val', image_f)
            image_labels = self.labels[image_key]
            
            if self.exp_type in ['gender', 'age', 'race']:
                image_labels['main_label'] = image_labels[self.exp_type]
            else:
                
                gender_namespace = image_labels['gender'][1]
                race_namespace = image_labels['race'][1]
                age_namespace = image_labels['age'][1]
                joint_key = (gender_namespace, race_namespace, age_namespace)
                joint_label = self.combination_label_dict[joint_key]
                image_labels['main_label'] = [joint_label, joint_key]
            
            datum = [image_fp, image_labels]
            self.data_list.append(datum)

    def combine_label_space(self):
        
        all_labels = [list(self.gender_labels.keys()),
                     list(self.race_labels.keys()),
                     list(self.age_labels.keys())]
        
        combination_labels = list(itertools.product(*all_labels))
        
        self.combination_label_dict = {}
        
        for label_id, combination in enumerate(combination_labels):
            self.combination_label_dict[combination] = label_id


    def parse_label_file(self):

        self.labels = {}

        age_ranges = []
        genders = []
        races = []
        '''
        {'Male', 'Female'}
        {'Black', 'Middle Eastern', 'Indian', 'White', 'Latino_Hispanic', 'East Asian', 'Southeast Asian'}
        {'30-39', '60-69', '3-9', '0-2', '10-19', '20-29', '40-49', 'more than 70', '50-59'}
        '''

        self.gender_labels = {'Male': 0, 'Female': 1}
        self.race_labels = {'Black':0, 'Middle Eastern':1, 'Indian':2, 'White':3,
                    'Latino_Hispanic':4, 'East Asian':5, 'Southeast Asian':6}
        self.age_labels = {'0-2': 0, '3-9':1, '10-19':2, '20-29':3, '30-39':4,
                            '40-49':5, '50-59':6, '60-69':7, 'more than 70': 8}

        self.gender_label_cts = {'Male': 0, 'Female': 0}
        self.race_label_cts = {'Black':0, 'Middle Eastern':0, 'Indian':0, 'White':0,
                    'Latino_Hispanic':0, 'East Asian':0, 'Southeast Asian':0}
        self.age_label_cts = {'0-2': 0, '3-9':0, '10-19':0, '20-29':0, '30-39':0,
                            '40-49':0, '50-59':0, '60-69':0, 'more than 70': 0}

        with open(self.label_f) as f:

            for idx,line in enumerate(f):
                if idx == 0:
                    continue
                line = line.strip('\n')
                img_name, age_range, gender, race, _ = line.split(',')
                if img_name in self.labels:
                    print ("Duplicate image, exiting")
                    exit()
                else:
                    self.labels[img_name] = {'gender': [self.gender_labels[gender], gender] ,
                                            'race': [self.race_labels[race], race],
                                            'age': [self.age_labels[age_range], age_range]}

                    self.gender_label_cts[gender] += 1
                    self.race_label_cts[race] += 1
                    self.age_label_cts[age_range] += 1



    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        datum = self.data_list[idx]
        '''
        image, labels = datum
        image = Image.open(image)

        gender_label, gender_name = labels['gender']
        age_label, age_name = labels['age']
        race_label, race_name = labels['race']
        main_label, main_name = labels['main_label']
    

        #if self.transform:
        #    image = self.transform(image)
        '''
        #return image, gender_label, age_label, race_label, main_label
        return datum

class RandomCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    def __call__(self, img):
        size = (min(img.size), min(img.size))
        # Only step forward along this edge if it's the long edge
        i = (0 if size[0] == img.size[0]
            else np.random.randint(low=0,high=img.size[0] - size[0]))
        j = (0 if size[1] == img.size[1]
            else np.random.randint(low=0,high=img.size[1] - size[1]))
        return transforms.functional.crop(img, i, j, size[0], size[1])

    def __repr__(self):
        return self.__class__.__name__


class CenterCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


class LoadDataset(Dataset):
    def __init__(self, dataset_name, data_path, train, download, resize_size, exp_type = None, hdf5_path=None, random_flip=False):
        super(LoadDataset, self).__init__()
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.train = train
        self.download = download
        self.resize_size = resize_size
        self.hdf5_path = hdf5_path
        self.random_flip = random_flip
        self.norm_mean = [0.5,0.5,0.5]
        self.norm_std = [0.5,0.5,0.5]
        self.exp_type = exp_type

        if self.hdf5_path is None:
            if self.dataset_name in ['cifar10', 'tiny_imagenet']:
                self.transforms = []
            elif self.dataset_name in ['imagenet', 'custom']:
                if train:
                    self.transforms = [RandomCropLongEdge(), transforms.Resize(self.resize_size)]
                else:
                    self.transforms = [CenterCropLongEdge(), transforms.Resize(self.resize_size)]
            elif self.dataset_name in ['fairface']:
                self.transforms = []
        else:
            self.transforms = [transforms.ToPILImage()]

        if random_flip:
            self.transforms += [transforms.RandomHorizontalFlip()]

        self.transforms += [transforms.ToTensor(), transforms.Normalize(self.norm_mean, self.norm_std)]
        self.transforms = transforms.Compose(self.transforms)
        self.load_dataset()


    def load_dataset(self):
        if self.dataset_name == 'cifar10':
            if self.hdf5_path is not None:
                print('Loading %s into memory...' % self.hdf5_path)
                with h5.File(self.hdf5_path, 'r') as f:
                    self.data = f['imgs'][:]
                    self.labels = f['labels'][:]
            else:
                self.data = CIFAR10(root=os.path.join('data', self.dataset_name),
                                    train=self.train,
                                    download=self.download)

        elif self.dataset_name == 'imagenet':
            if self.hdf5_path is not None:
                print('Loading %s into memory...' % self.hdf5_path)
                with h5.File(self.hdf5_path, 'r') as f:
                    self.data = f['imgs'][:]
                    self.labels = f['labels'][:]
            else:
                mode = 'train' if self.train == True else 'valid'
                root = os.path.join('data','ILSVRC2012', mode)
                self.data = ImageFolder(root=root)

        elif self.dataset_name == "tiny_imagenet":
            if self.hdf5_path is not None:
                print('Loading %s into memory...' % self.hdf5_path)
                with h5.File(self.hdf5_path, 'r') as f:
                    self.data = f['imgs'][:]
                    self.labels = f['labels'][:]
            else:
                mode = 'train' if self.train == True else 'valid'
                root = os.path.join('data','TINY_ILSVRC2012', mode)
                self.data = ImageFolder(root=root)

        elif self.dataset_name == "custom":
            if self.hdf5_path is not None:
                print('Loading %s into memory...' % self.hdf5_path)
                with h5.File(self.hdf5_path, 'r') as f:
                    self.data = f['imgs'][:]
                    self.labels = f['labels'][:]
            else:
                mode = 'train' if self.train == True else 'valid'
                root = os.path.join('data','CUSTOM', mode)
                self.data = ImageFolder(root=root)
        
        elif self.dataset_name == 'fairface':
            self.data = FairFaceDataset(self.data_path, 
                                        exp_type = self.exp_type,
                                        train = self.train,
                                        transform = self.transforms)
        
        else:
            raise NotImplementedError


    def __len__(self):
        if self.hdf5_path is not None:
            num_dataset = self.data.shape[0]
        else:
            num_dataset = len(self.data)
        return num_dataset


    def __getitem__(self, index):
        if self.hdf5_path is None:
            img, label = self.data[index]
            img = Image.open(img)
            img = self.transforms(img)
        else:
            img, label = np.transpose(self.data[index], (1,2,0)), int(self.labels[index])
            img = self.transforms(img)
        
        gender_label, gender_name = label['gender']
        age_label, age_name = label['age']
        race_label, race_name = label['race']
        main_label, main_name = label['main_label']
 
        return img, [gender_label, age_label, race_label, main_label]
