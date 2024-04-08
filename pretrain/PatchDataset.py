from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
from tqdm import tqdm
import torch

class PatchDataset(data.Dataset):
    def __init__(self, norms_path, phase='train'):
        files = os.listdir(norms_path)
        if phase == 'train':
            files = files[:80]
        else:
            files = files[80:]

        all_images = []
        for file in tqdm(files):
            images = np.load(pjoin(norms_path, file), allow_pickle=True).item()['norms']
            all_images.append(images)

        all_images = np.concatenate(all_images, axis=0)
        print(f'Total {all_images.shape[0]} images in {phase} set')
        self.all_images = np.transpose(all_images, (0, 3, 1, 2))
        self.mean = np.array([180.87767965, 147.45242127, 172.7674556])
        self.std = np.array([47.7491585, 61.73678167, 46.65056828])


    def __len__(self):
        return self.all_images.shape[0]

    def __getitem__(self, index):
        return self.all_images[index], self.all_images[index]

    def collate_fn(self, batch):
        images = []
        gts = []
        for (x, y) in batch:
            images.append(x)
            gts.append(y)

        images = np.array(images)
        gts = np.array(gts)
        
        bs, c, h, w = images.shape
        
        assert c == 3 and h == 256 and w == 256

        images = (images - self.mean.reshape(1, -1, 1, 1)) / self.std.reshape(1, -1, 1, 1)
        gts = (gts - self.mean.reshape(1, -1, 1, 1)) / self.std.reshape(1, -1, 1, 1)
        return torch.Tensor(images), torch.Tensor(gts)

