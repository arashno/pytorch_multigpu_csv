import pandas as pd
import os
import numpy as np
import torch
import random

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import *
from PIL import Image, ImageStat

class CSVDataset(Dataset):

    def __init__(self, input_file, delimiter, raw_size, processed_size, batch_size, num_threads, path_prefix, is_training, shuffle=False, inference_only= False):
        self.input_file = input_file
        self.raw_size = raw_size
        self.processed_size = processed_size
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.shuffle = shuffle
        self.path_prefix = path_prefix
        self.inference_only = inference_only
        self.info = pd.read_csv(input_file, delimiter = delimiter)
        self.image_names = self.info.iloc[:,0].tolist()
        if not inference_only:
            self.labels = torch.tensor(self.info.iloc[:,1].tolist())
        self.is_training = is_training
        self.transform = self.build_transforms()

    def load(self):
        return DataLoader(self, batch_size = self.batch_size, shuffle = self.shuffle, num_workers = self.num_threads)

    def build_transforms(self):
        transform_list = []
        transform_list.append(Resize(self.raw_size[0:2]))
        if self.is_training:
            transform_list.append(RandomCrop(self.processed_size[0:2]))
            transform_list.append(RandomHorizontalFlip())
            transform_list.append(ColorJitter())
            transform_list.append(RandomRotation(20))
        else:
            transform_list.append(transforms.CenterCrop(self.processed_size[0:2]))
        transform_list.append(ToTensor())
        mean, std = self.calc_mean_std()
        transform_list.append(Normalize(mean, std))
        return Compose(transform_list)

    def calc_mean_std(self):
        cache_file = '.' + self.input_file + '_meanstd' + '.cache'
        if not os.path.exists(cache_file):
            print('Calculating Mean and Std')
            means = np.zeros((3))
            stds = np.zeros((3))
            sample_size = min(len(self.image_names), 10000)
            for i in range(sample_size):
                img_name = os.path.join(self.path_prefix, random.choice(self.image_names))
                img = Image.open(img_name).convert('RGB')
                stat = ImageStat.Stat(img)
                means += np.array(stat.mean) / 255.0
                stds += np.array(stat.stddev) / 255.0
            means = means / sample_size
            stds = stds / sample_size
            np.savetxt(cache_file, np.vstack((means, stds)))
        else:
            print('Load Mean and Std from ' + cache_file)
            contents = np.loadtxt(cache_file)
            means = contents[0,:]
            stds = contents[1,:]

        return means, stds

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.path_prefix, self.image_names[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            sample = self.transform(image)
        if not self.inference_only:
            return sample, self.labels[idx], self.image_names[idx]
        else:
            return sample, self.image_names[idx]
