import torch
from torch.utils.data import DataLoader
from PatchDataset import PatchDataset 
from encdec import EncDec
from tqdm import tqdm
from torch.optim import lr_scheduler
import os
from os.path import join as pjoin
import time
import json
import numpy as np
import shutil
from torch.utils.tensorboard import SummaryWriter

def get_dataloader(norms_path='../dataset/TCGA/normed_lowres', phase='train', batch_size=32):
    dataset = PatchDataset(norms_path, phase)
    if phase == 'train':
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True, collate_fn=dataset.collate_fn)
    else:
        return DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8, drop_last=False, collate_fn=dataset.collate_fn)


mean = np.array([0.70932423, 0.57824479, 0.67751943])
std = np.array([0.1872516 , 0.24210503, 0.18294341])

def main(
    epochs_num=80,
    device='cuda',
    hidden_channels=32,
    log_interval = 100,
    eval_interval=10,
    lr=1e-3,
    ckpt_path='./runs',
    batch_size=32,
    resume_from=None,
    start_epoch=0
    ):
    #--> get dataloader
    train_loader = get_dataloader(phase='train', batch_size=batch_size)
    val_loader = get_dataloader(phase='val', batch_size=batch_size)
    #--> get models
    model = EncDec(hidden_channels)
    model.to(device)
    #--> set losses
    loss_fn = torch.nn.MSELoss()
    #--> for optims
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    #--> initial logs
    step = 0
    os.makedirs(ckpt_path, exist_ok=True)
    current_time = time.strftime("%m-%d-%H-%M", time.localtime())
    os.makedirs(pjoin(ckpt_path, current_time))
    with open(pjoin(ckpt_path, current_time, 'config.json'), mode='w') as f:
        config = {
            'epochs_num': epochs_num,
            'hidden_channels': hidden_channels,
            'log_interval': log_interval,
            'eval_interval': eval_interval,
            'lr': lr,
            'batch_size': batch_size
        }
        json.dump(config, f, indent=4)    
    writer = SummaryWriter(log_dir='./logs')
    losses = {}
    shutil.copyfile('pretrain/encdec.py', pjoin(ckpt_path, current_time, 'encdec.py'))
    
    #--> eval function
    def eval():
        model.eval()
        loss = 0
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss += loss_fn(pred, y).detach().item()
        tqdm.write('########### EVAL ###########')
        tqdm.write(f'Epoch: {epoch}, Loss on val set: {loss / len(val_loader)}')
        losses[epoch] = loss / len(val_loader)
        writer.add_scalar("eval/loss", loss / len(val_loader), step)
        writer.add_image('eval/pred', pred[0].detach().to('cpu') * std.reshape(-1, 1, 1) + mean.reshape(-1, 1, 1), step)
        writer.add_image('eval/true', y[0].detach().to('cpu') * std.reshape(-1, 1, 1) + mean.reshape(-1, 1, 1), step)
    
    
    #--> start training
    if resume_from is not None:
        state_dict = torch.load(pjoin(resume_from, 'model.pt'))
        model.load_state_dict(state_dict)
        step = len(train_loader) * (start_epoch - 1)
        opt_state_dict = torch.load(pjoin(resume_from, 'optimizer.pt'))
        optimizer.load_state_dict(opt_state_dict)
        sch_state_dict = torch.load(pjoin(resume_from, 'scheduler.pt'))
        scheduler.load_state_dict(sch_state_dict)
    
    model.train()
    for epoch in tqdm(range(start_epoch, epochs_num)):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            
            optimizer.zero_grad()
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            if step % log_interval == 0:
                tqdm.write(f'Epoch: {epoch}, Global Step: {step}, Loss: {loss.detach().item()}')
                writer.add_scalar("train/loss", loss.detach().item(), step)
                writer.add_image('train/pred', pred[0].detach().to('cpu') * std.reshape(-1, 1, 1) + mean.reshape(-1, 1, 1), step)
                writer.add_image('train/true', y[0].detach().to('cpu') * std.reshape(-1, 1, 1) + mean.reshape(-1, 1, 1), step)
            step += 1
        
        scheduler.step()
        if epoch % eval_interval == 0:
            tqdm.write('Eval Start')
            eval()
            os.makedirs(pjoin(ckpt_path, current_time, str(epoch)))
            torch.save(model.state_dict(), pjoin(ckpt_path, current_time, str(epoch), 'model.pt'))
            torch.save(optimizer.state_dict(), pjoin(ckpt_path, current_time, str(epoch), 'optimizer.pt'))
            torch.save(scheduler.state_dict(), pjoin(ckpt_path, current_time, str(epoch), 'scheduler.pt'))
            model.train()
    
    #--> eval and save the final model
    eval()
    os.makedirs(pjoin(ckpt_path, current_time, str(epoch)), exist_ok=True)
    torch.save(model.state_dict(), pjoin(ckpt_path, current_time, str(epoch), 'model.pt'))
    torch.save(optimizer.state_dict(), pjoin(ckpt_path, current_time, str(epoch), 'optimizer.pt'))
    torch.save(scheduler.state_dict(), pjoin(ckpt_path, current_time, str(epoch), 'scheduler.pt'))
    best_loss = 10000
    best_epoch = 0
    for key, values in losses.items():
        if values <  best_loss:
            best_loss = values
            best_epoch = key
    print(f"Best performance: {best_loss}, Epoch: {best_epoch}")

if __name__ == '__main__':
    main()