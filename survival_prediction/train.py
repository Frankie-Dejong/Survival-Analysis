# import pandas as pd
import torch.nn as nn
import torch
import torch.nn.init as init
from torch.optim import lr_scheduler
from tqdm import tqdm
import random
import numpy as np
from utils import *
from scipy.stats import weibull_min
from Models import CancerPredictModel
from CancerDataset import CancerDataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
 
def train_model(
    dataset: CancerDataset,
    test_set: CancerDataset,
    model: CancerPredictModel,
    epochs: int,
    lr: float,
):  
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.8)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    model.train()
    writer = SummaryWriter('./logs_II')
    step = 0
    for epoch in tqdm(range(epochs)):
        
        if epoch % 10 == 0:
            test_loss = test(test_set)
            writer.add_scalar('eval/loss', test_loss, epoch)
            model.train()
        
        
        for bags, real_days, vital_status, ids  in dataset:
            for i in range(dataset.batch_size):
                bag = torch.Tensor(bags[i]).to(device)
                real_day = torch.Tensor(real_days[i]).to(device)

                vs = torch.Tensor(vital_status[i]).to(device)
            
                alpha, beta = model(bag)
                # alpha, beta = map_weibull_param(alpha, beta)
                y_true = torch.concat((real_day.unsqueeze(0), vs.unsqueeze(0)))
                
                optimizer.zero_grad()
                if vital_status[i] == 1:
                    loss = weibull_log_likelihood(y_true, alpha, beta)
                else:
                    loss = 0.3 * weibull_log_likelihood(y_true, alpha, beta)
                loss.backward()
                optimizer.step()
                
                step += 1
                
                if step % 500 == 0:
                    # tqdm.write(f'loss on train set: {loss.detach().cpu()}')
                    writer.add_scalar('train/loss', loss.detach().cpu(), step)
        dataset.on_epoch_end()
        # scheduler.step()
        
        if VERBOSE:
            tqdm.write(f'LOSS: {loss.item()}, day: {real_day.item()}, VS: {vs.item()}')
        
            

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def test(dataset):
    model.eval()
    tot = 0
    loss = 0
    for bags, real_days, vital_status, ids in dataset:
        for i in range(dataset.batch_size):
            bag = torch.Tensor(bags[i]).to(device)
            real_day = torch.Tensor(real_days[i]).to(device)
            vs = torch.Tensor(vital_status[i]).to(device)
            
            y_true = torch.concat((real_day.unsqueeze(0), vs.unsqueeze(0)))
            alpha, beta = model(bag)
            # alpha, beta = map_weibull_param(alpha, beta)
            
            loss += weibull_log_likelihood(y_true, alpha, beta)
            
            tot += 1
            

    tqdm.write(f'Loss on val set: {loss.detach().cpu() / tot}')
    return loss.detach().cpu() / tot


if __name__ == "__main__":
    import yaml
    import os
    
    seed_everything(1225)

    cfg = yaml.safe_load(open("configuration/tcga_cfg.yml", "r"))
    train_set = CancerDataset(feat_path=cfg['feature_path'], anno_dir=cfg["anno_path"], batch_size=8, loader='numpy')
    test_set = CancerDataset(feat_path=cfg['feature_path'], anno_dir=cfg["anno_path"], batch_size=1, phase='test', loader='numpy')
    model = CancerPredictModel()
    device = torch.device(cfg["device"])
    model.to(device)

    def init_linear(m):
        if isinstance(m, nn.Linear):
            init.kaiming_uniform_(m.weight)
            init.constant_(m.bias, 0)
    model.apply(init_linear)
    VERBOSE = False
    train_model(train_set, test_set, model, epochs=250, lr=1e-3)
    test(test_set)
    torch.save(model.state_dict(), './ckpt.pt')
