import torch 
import logging
import os

from torch import Tensor, nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy 
from model.chmg.IPMAE import InterpolateMAE
from dataloaders.get_data import get_dataset_loader
from eval.eval_utils import ipmae_evaluate
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('log/IPMAE/')


def update_loss_plot(train_losses, valid_losses, max_epoch):
    plt.clf()  # Clear the previous plot
    plt.plot(range(0, len(train_losses)), 
             train_losses, marker='o', color='blue', 
             markersize = 1,label='train')
    plt.plot(range(0, len(valid_losses)), 
             valid_losses, marker='o', 
             markersize=1, color='orange',label='val')
    plt.yscale('log')
    plt.yticks([0.0001, 0.001, 0.01,0.1, 1])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim((0, max_epoch))
    #plt.ylim((0, 0.5))
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.pause(0.1)


def train():
    cfg = OmegaConf.load('./config/train/train_IPMAE.yaml')
    model = InterpolateMAE(cfg=cfg)
    if os.path.exists(cfg.CHECKPOINTS):
        model.load_state_dict(torch.load(cfg.CHECKPOINTS))
    model.TextEncoder.eval()
    humanml_train_loader = get_dataset_loader(
        name=cfg.DATALOADER.NAME,
        batch_size=cfg.DATALOADER.batch_size,
        split='train',
        num_frames=None,
        hml_mode='train')
    
    humanml_valid_loader = get_dataset_loader(
        name=cfg.DATALOADER.NAME,
        batch_size=cfg.DATALOADER.batch_size,
        split='test',
        num_frames=None,
        hml_mode='eval')

    humanml_gt_loader = get_dataset_loader(
        name=cfg.DATALOADER.NAME,
        batch_size=cfg.DATALOADER.batch_size,
        split='test',
        num_frames=None,
        hml_mode='gt')
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.TRAIN.lr)
    
    train_loop(cfg,
               model, 
               humanml_train_loader,
               humanml_valid_loader,
               humanml_gt_loader,
               optimizer=optimizer,
               epochs=cfg.TRAIN.EPOCHS)
    

def train_loop(cfg:DictConfig,
               model:InterpolateMAE, 
               train_dataloader:DataLoader,
               valid_dataloader:DataLoader,
               gt_dataloader:DataLoader,
               optimizer:torch.optim.Optimizer,
               epochs:int=500
               ):
    
    epoch_train_losses = []
    epoch_valid_losses = []
    for e in range(epochs):
        batch_train_losses = []
        batch_valid_losses = []
        
        model.train()
        
        train_bar = tqdm(total=len(train_dataloader))
        
        
        for batch in train_dataloader:
            torch.cuda.empty_cache()
            model.train()
            X = batch[0].squeeze(2).transpose(-1,-2).to(model.device)
            lengths = batch[1]['y']['lengths'].to(model.device)
            text = batch[1]['y']['text']
            batch_train_loss = model.compute_loss(
                X=X,
                text_cond=text,
                lengths=lengths
            )
            
            optimizer.zero_grad()
            batch_train_loss.backward()
            optimizer.step()
            batch_train_losses.append(batch_train_loss.cpu().detach().numpy())
            prefix = f"Epoch{e}_{train_bar.n}/{len(train_dataloader)}. Loss:{batch_train_loss:.4f}"
            train_bar.set_description(prefix)
            train_bar.update()
            
            del X, batch, lengths, batch_train_loss
            torch.cuda.empty_cache()
        
        model.eval()
        valid_bar = tqdm(total=len(valid_dataloader))
        
        for batch in valid_dataloader:
            torch.cuda.empty_cache()
            X = batch[0].squeeze(2).transpose(-1,-2).to(model.device)
            lengths = batch[1]['y']['lengths'].to(model.device)
            text = batch[1]['y']['text']
            batch_valid_loss = model.compute_loss(
                X=X,
                text_cond=text,
                lengths=lengths
            )
            batch_valid_losses.append(batch_valid_loss.cpu().detach().numpy())
            prefix = f"Epoch{e}_{valid_bar.n}/{len(valid_dataloader)}. Loss:{batch_valid_loss:.4f}"
            valid_bar.set_description(prefix)
            valid_bar.update()
            del X, batch, lengths, batch_valid_loss
            torch.cuda.empty_cache()
            
        epoch_train_loss = numpy.stack(batch_train_losses).mean(0)
        epoch_valid_loss = numpy.stack(batch_valid_losses).mean(0)
        
        epoch_train_losses.append(epoch_train_loss)
        epoch_valid_losses.append(epoch_valid_loss)
        writer.add_scalar('Train/Loss', epoch_train_loss.item(), e)
        writer.add_scalar('Valid/Loss', epoch_valid_loss.item(), e)

        update_loss_plot(epoch_train_losses,
                         epoch_valid_losses,
                         max_epoch=epochs)
        
        if epoch_valid_loss <= min(epoch_valid_losses):
            torch.save(model.state_dict(),cfg.CHECKPOINTS)
            
        if e%10==0:

            log_path = cfg.LOG.path
            log_file = open(log_path, 'a')
            
            from datetime import datetime
            print('========================================', file=log_file)
            print('Time',datetime.now(),file=log_file)
            print('Evaluation epoch',e,file=log_file)
            ipmae_evaluate(model=model,
                           evaldataloader=valid_dataloader,
                           gt_loader=gt_dataloader,
                           FILE=log_file,
                           datasetname=cfg.DATALOADER.NAME,
                           batch_size=cfg.DATALOADER.batch_size)
    plt.savefig('./ipmae_training_loss_plot.png')
            
if __name__ == '__main__':
    train()