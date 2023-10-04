import torch 
import logging
import os

from torch import Tensor, nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
import numpy 
from model.sdm.sdm import SDM
from dataloaders.get_data import get_dataset_loader
from eval.eval_utils import T2MEvaluator
from torch.utils.tensorboard import SummaryWriter


import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def train():
    cfg = OmegaConf.load('./config/train/train_KFSDM.yaml')
    print('Training at the stage of:', cfg.stage)
    model = SDM(cfg=cfg)
    if cfg.stage == 'vae':
        checkpoints = 'checkpoints/' + cfg.EXNAME + '_vae.pth'
        if os.path.exists(checkpoints):
            model.vae = torch.load(checkpoints)
            print('Load Pretrained VAE from:', checkpoints)
    elif cfg.stage == 'diffusion':
        vae_checkpoints = 'checkpoints/' + cfg.EXNAME + '_vae.pth'
        dn_checkpoints = 'checkpoints/' + cfg.EXNAME + '_dn.pth'
        if os.path.exists(vae_checkpoints):
            model.vae = torch.load(vae_checkpoints)
            print('Load Pretrained VAE from:', vae_checkpoints)
        else:
            print('No pretrained VAE, will be created')
        if os.path.exists(dn_checkpoints):
            model.denoiser = torch.load(dn_checkpoints)
            print('Load Pretrained Denoiser from:', dn_checkpoints)
        else:
            print('No pretrained Denoiser, will be created')
        
    else:
        raise NotImplementedError
            
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
               model:SDM, 
               train_dataloader:DataLoader,
               valid_dataloader:DataLoader,
               gt_dataloader:DataLoader,
               #optimizer:torch.optim.Optimizer,
               epochs:int=2000,
               ):
    


    epoch_train_losses = []
    epoch_valid_losses = []
    train_bar = tqdm(total=len(train_dataloader))
    valid_bar = tqdm(total=len(valid_dataloader))
    for e in range(epochs):
        batch_train_losses = []
        batch_valid_losses = []
        batch_train_kl_losses = []
        batch_valid_kl_losses = []
        
        model.train()
        
        if cfg.stage == 'vae':
            optimizer = torch.optim.AdamW(model.vae.parameters(), lr=cfg.TRAIN.lr)

            for batch in train_dataloader:
                torch.cuda.empty_cache()
                model.vae.train()
                X = batch[0].squeeze(2).transpose(-1,-2).to(model.device)
                lengths = batch[1]['y']['lengths'].to(model.device)
                text = batch[1]['y']['text']
                
                batch_train_loss = model.compute_loss(
                    motions=X, text=text,lengths=lengths
                )
                
                batch_train_kl_loss = batch_train_loss[-1].cpu().detach().numpy()
                batch_train_kl_losses.append(batch_train_kl_loss)
                batch_train_loss = batch_train_loss[0]
                
                optimizer.zero_grad()
                batch_train_loss.backward()
                optimizer.step()
                batch_train_loss = batch_train_loss.cpu().detach().numpy()
                batch_train_losses.append(batch_train_loss)
                prefix = f"Epoch{e}_{train_bar.n%len(train_dataloader)}/{len(train_dataloader)}. \
                    Loss:{batch_train_loss:.5f}. KL Loss:{batch_train_kl_loss:.5f}. \
                    Rec Loss:{(batch_train_loss - batch_train_kl_loss * cfg.VAE.beta_kl):.5f}"
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
                #X = model.get_keyframes(X)
                lengths = batch[1]['y']['lengths'].to(model.device)
                text = batch[1]['y']['text']
                batch_valid_loss = model.compute_loss(
                    motions=X,
                    text=text,
                    lengths=lengths
                )
                
                batch_valid_kl_loss = batch_valid_loss[-1].cpu().detach().numpy()
                batch_valid_kl_losses.append(batch_valid_kl_loss)
                batch_valid_loss = batch_valid_loss[0].cpu().detach().numpy()
                
                batch_valid_losses.append(batch_valid_loss)
                prefix = f"Epoch{e}_{valid_bar.n%len(valid_dataloader)}/{len(valid_dataloader)}. \
                    Loss:{batch_valid_loss:.5f}. KL Loss:{batch_valid_kl_loss:.5f}. \
                    Rec Loss:{(batch_valid_loss-batch_valid_kl_loss * cfg.VAE.beta_kl):.5f}"            
                valid_bar.set_description(prefix)
                valid_bar.update()
                del X, batch, lengths, batch_valid_loss
                torch.cuda.empty_cache()
                
            epoch_train_loss = numpy.stack(batch_train_losses).mean(0)
            epoch_valid_loss = numpy.stack(batch_valid_losses).mean(0)
            
            epoch_train_kl_loss = numpy.stack(batch_train_kl_losses).mean(0)
            epoch_valid_kl_loss = numpy.stack(batch_valid_kl_losses).mean(0)
            
            epoch_train_losses.append(epoch_train_loss)
            epoch_valid_losses.append(epoch_valid_loss)
            

            writer.add_scalar('Train/Loss', epoch_train_loss.item(), e)
            writer.add_scalar('Valid/Loss', epoch_valid_loss.item(), e)
            writer.add_scalar('Train/RecLoss', epoch_train_loss.item() - epoch_train_kl_loss.item(),e)
            writer.add_scalar('Train/KLLoss', epoch_train_kl_loss.item(), e)
            writer.add_scalar('Valid/KLLoss', epoch_valid_kl_loss.item(), e)
            writer.add_scalar('Valid/RecLoss', epoch_valid_loss.item() - epoch_valid_kl_loss.item(),e)

        elif cfg.stage  == 'diffusion':
            optimizer = torch.optim.AdamW(model.denoiser.parameters(), lr=cfg.TRAIN.lr)
            model.denoiser.train()
            model.text_encoder.eval()
            model.vae.eval()
            for para in model.text_encoder.parameters():
                para.requires_grad = False 
            for para in model.vae.parameters():
                para.requires_grad = False
                
            for batch in train_dataloader:
                torch.cuda.empty_cache()
                
                X = batch[0].squeeze(2).transpose(-1,-2).to(model.device)
                lengths = batch[1]['y']['lengths'].to(model.device)
                text = batch[1]['y']['text']
                batch_train_loss = model.compute_loss(
                    motions=X,
                    text=text,
                    lengths=lengths
                )
                                
                optimizer.zero_grad()
                batch_train_loss.backward()
                optimizer.step()
                batch_train_loss = batch_train_loss.cpu().detach().numpy()
                batch_train_losses.append(batch_train_loss)
                prefix = f"Epoch{e}_{train_bar.n%len(train_dataloader)}/{len(train_dataloader)}.\
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
                X = batch[0].squeeze(2).transpose(-1,-2).to(model.device)
                #X = model.get_keyframes(X)
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
            
            if eval_result['FID'] <= 0.30:
                torch.save(model.denoiser, 'best_dn_'+str(eval_result['FID'])+'.pth')
            if eval_result['R_precision_top_3'] >= 0.78:
                torch.save(model.denoiser, 'best_dn_'+str(eval_result['R_precision_top_3'])+'.pth')
            
            del t2m_evaluator
            model.save_model()
            
if __name__ == '__main__':
    train()