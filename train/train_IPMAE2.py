import torch 
import logging
import os

from torch import Tensor, nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
import numpy 
from model.chmg.IPMAE2 import InterpolateMAE
from dataloaders.get_data import get_dataset_loader
from eval.eval_utils import T2MEvaluator
from torch.utils.tensorboard import SummaryWriter


import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def train():
    cfg = OmegaConf.load('./config/train/train_IPMAE2.yaml')
    model = InterpolateMAE(cfg=cfg)
    ipmae_checkpoint = 'checkpoints/'+cfg.EXNAME+'_mae.pth'
    if os.path.exists(ipmae_checkpoint):
        model = torch.load(ipmae_checkpoint)
        print('Resume from checkpoints:', ipmae_checkpoint)
        
        
    #model.TextEncoder.eval()
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
    
    
    writer = SummaryWriter('log/'+cfg.EXNAME)
    
    train_loop(cfg=cfg,
               writer=writer,
               model=model,
               train_dataloader=humanml_train_loader,
               valid_dataloader=humanml_valid_loader,
               gt_dataloader=humanml_gt_loader,
               epochs=3000)
    


def train_loop(cfg:DictConfig,
               writer:SummaryWriter,
               model:InterpolateMAE, 
               train_dataloader:DataLoader,
               valid_dataloader:DataLoader,
               gt_dataloader:DataLoader,
               epochs:int=2000,
               ):
    


    epoch_train_losses = []
    epoch_valid_losses = []
    train_bar = tqdm(total=len(train_dataloader))
    valid_bar = tqdm(total=len(valid_dataloader))
    for e in range(epochs):
        batch_train_losses = []
        batch_valid_losses = []
        
        model.train()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.TRAIN.lr)

        for batch in train_dataloader:
                torch.cuda.empty_cache()
                model.train()
                model.text_encoder.eval()
                for para in model.text_encoder.parameters():
                    para.requires_grad = False
                
                X = batch[0].squeeze(2).transpose(-1,-2).to(model.device)
                lengths = batch[1]['y']['lengths'].to(model.device)
                text = batch[1]['y']['text']
                
                batch_train_loss = model.compute_loss(
                    motions=X, text=text,lengths=lengths
                )
                                
                optimizer.zero_grad()
                batch_train_loss.backward()
                optimizer.step()
                batch_train_loss = batch_train_loss.cpu().detach().numpy()
                batch_train_losses.append(batch_train_loss)
                prefix = f"Epoch{e}_{train_bar.n%len(train_dataloader)}/{len(train_dataloader)}. \
                    Loss:{batch_train_loss:.5f}."
                train_bar.set_description(prefix)
                train_bar.update()
                
                del X, batch, lengths, batch_train_loss
                torch.cuda.empty_cache()
        if e % 5 != 0:
            continue
        model.eval()
            
            
        for batch in valid_dataloader:
                torch.cuda.empty_cache()
                X = batch[0].squeeze(2).transpose(-1,-2).to('cuda')
                lengths = batch[1]['y']['lengths'].to(model.device)
                text = batch[1]['y']['text']
                batch_valid_loss = model.compute_loss(
                    motions=X,
                    text=text,
                    lengths=lengths
                )
                
                batch_valid_loss = batch_valid_loss.cpu().detach().numpy()
                
                batch_valid_losses.append(batch_valid_loss)
                prefix = f"Epoch{e}_{valid_bar.n%len(valid_dataloader)}/{len(valid_dataloader)}. \
                    Loss:{batch_valid_loss:.5f}."            
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
            
        if e%20==0:
            t2m_evaluator = T2MEvaluator(cfg=cfg,
                                        eval_dataset=valid_dataloader.dataset,
                                        model=model)
            eval_result = t2m_evaluator.evaluate()
            print(eval_result)
            writer.add_scalar('MM-Distance', eval_result['Matching_score'],e)
            writer.add_scalar('R-top1', eval_result['R_precision_top_1'],e)
            writer.add_scalar('R-top2', eval_result['R_precision_top_2'],e)
            writer.add_scalar('R-top3', eval_result['R_precision_top_3'],e)
            writer.add_scalar('FID', eval_result['FID'],e)
            writer.add_scalar('Diversity', eval_result['Diversity'],e)
            
            # if eval_result['FID'] <= 0.30:
            #     torch.save(model.denoiser, 'best_dn_'+str(eval_result['FID'])+'.pth')
            # if eval_result['R_precision_top_3'] >= 0.78:
            #     torch.save(model.denoiser, 'best_dn_'+str(eval_result['R_precision_top_3'])+'.pth')
            
            del t2m_evaluator
            model.save_model()
            
if __name__ == '__main__':
    train()