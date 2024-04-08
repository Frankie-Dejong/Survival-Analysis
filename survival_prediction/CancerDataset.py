import torch
from torch.utils.data import Dataset
from utils import load_tsv

import random

from os.path import join as pjoin
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import torch.nn.functional as F


def rms(x):
    return torch.sqrt(F.mse_loss(x, torch.zeros_like(x)))

class CancerDataset(Dataset):
    def __init__(self, feat_path, anno_dir, dataset='TCGA', batch_size=8, shuffle=True, verbose=False, phase='train', loader='numpy'):
        self.anno_dir = anno_dir
        self.data_dict = {}
        self.batch_size = batch_size
        self.data = {}
        self.available = []
        self._verbose = verbose

        if phase == 'train':
            for file in tqdm(os.listdir(feat_path)[:-50]):
                if loader == 'numpy':
                    self.data[file.split('.')[0]] = np.load(pjoin(feat_path, file), allow_pickle=True).item()
                else:
                    self.data[file.split('.')[0]] = torch.load(pjoin(feat_path, file))
                self.available.append(file.split('.')[0])
                if verbose:
                    print(f"Loading file {file}")
        else:
            for file in tqdm(os.listdir(feat_path)[-100:]):
                if loader == 'numpy':
                    self.data[file.split('.')[0]] = np.load(pjoin(feat_path, file), allow_pickle=True).item()
                else:
                    self.data[file.split('.')[0]] = torch.load(pjoin(feat_path, file))
                self.available.append(file.split('.')[0])
                if verbose:
                    print(f"Loading file {file}")
        print(f"Total {len(self.data)} files loaded") 
        self.shuffle = shuffle

        days = []

        if dataset == 'TCGA':    
            anno_coad = load_tsv(pjoin(anno_dir, 'TCGA-COAD.survival.tsv'))
            anno_read = load_tsv(pjoin(anno_dir, 'TCGA-READ.survival.tsv'))
            anno_tcga = pd.concat([anno_coad, anno_read]).set_index('_PATIENT')
            
            for _, row in anno_tcga.iterrows():
                name = row.name
                sample = row['sample']
                vital_status = row['OS']
                real_days = row['OS.time']

                if sample not in self.available:
                    continue
                
                self.data_dict[sample] = {
                        'sample': sample, 
                        'vital_status': vital_status,
                        'real_days': real_days
                    }
                days.append(real_days)
                if verbose:
                    print(f"Adding sample {sample}") 
            print(f"Total {len(self.data_dict)} names")
        else:
            raise NotImplementedError("Unsupported dataset: {}".format(dataset))
        
        self.mean = 885.0
        self.std = 750.125244140625
        self.rms = 1159.7108154296875
        print(f"Mean: {self.mean}, Std: {self.std}, Rms: {self.rms}")
        self.index_list = np.arange(len(self.data_dict) // self.batch_size)
        
    def __len__(self):
        return len(self.data_dict) // self.batch_size
    
    def __getitem__(self, idx):
        index = self.index_list[idx]
        patient_ids = list(self.data_dict.keys())[index * self.batch_size: (index + 1) * self.batch_size]
        return self._generate_batch(patient_ids)
    
    def on_epoch_end(self):
        self.index_list = np.arange(len(self.data_dict) // self.batch_size)
        if self.shuffle:
            np.random.shuffle(self.index_list)
     
        
    def _generate_batch(self, patient_ids):
        bags = []
        real_days = []
        vital_status = []
        
        for patient_id in patient_ids:
            if self._verbose:
                print(f"patient id: {patient_id}")
            bags.append(self.data[patient_id]["features"])
            real_days.append(self.data_dict[patient_id]['real_days']) 
            vital_status.append(self.data_dict[patient_id]['vital_status']) 
        
        real_days = torch.tensor(real_days, dtype=torch.float)
        vital_status = torch.tensor(vital_status, dtype=torch.int64)
        
        return  bags, real_days / self.rms, vital_status, patient_ids
