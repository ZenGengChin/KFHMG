import torch 

import os 


from typing import List, Optional
from torch import nn, Tensor
from omegaconf import DictConfig
from transformers import RobertaModel, RobertaTokenizer
from transformers import T5Model, T5Tokenizer
from transformers import ElectraModel, ElectraTokenizer

from model.chmg.ctrm import CondResTransformerEncoder

from model.sdm.sdm_textencoder import KFTextEncoder


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        
        # Create a positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(device=x.device)

class InterpolateMAE(nn.Module):
    def __init__(self,
                 cfg:DictConfig) -> None:
        super().__init__()
        
        self.cfg = cfg
        self.device = cfg.IPMAE.device
        
        self.MotionInProj = nn.Linear(cfg.IPMAE.d_input, 
                                    cfg.IPMAE.TRMDECODER.d_model).to(self.device)
        self.MotionOutProj = nn.Linear(cfg.IPMAE.TRMDECODER.d_model,
                                       cfg.IPMAE.d_input).to(self.device)
        
        self.PE = PositionalEncoding(d_model=cfg.IPMAE.TRMDECODER.d_model,
                                     max_seq_len=cfg.IPMAE.max_motion_len)
        if self.cfg.IPMAE.arch == 'trans_enc':
            self.TRMEncoderLayer = nn.TransformerEncoderLayer(
                d_model=cfg.IPMAE.TRMDECODER.d_model,
                nhead=cfg.IPMAE.TRMDECODER.nhead,
                dim_feedforward=cfg.IPMAE.TRMDECODER.d_ffn,
                batch_first=cfg.IPMAE.TRMDECODER.batch_first,
                dropout=cfg.IPMAE.TRMDECODER.dropout,
                device=cfg.IPMAE.device)
            
            self.TRMEncoder = CondResTransformerEncoder(
                encoder_layer=self.TRMEncoderLayer,
                num_layers=cfg.IPMAE.TRMDECODER.nlayer,
                condition_plus=True
            )
        else:
            raise NotImplementedError
        
        self.mask_method = cfg.IPMAE.MASK.method
        self.mask_ratio = cfg.IPMAE.MASK.ratio
        self.nhead = cfg.IPMAE.TRMDECODER.nhead

        self.text_encoder = KFTextEncoder(cfg=cfg)
            
        self.TextProj = nn.Linear(cfg.SDMTEXTENCODER.d_model, 
                                  cfg.IPMAE.TRMDECODER.d_model).to(self.device)
        self.loss_fn = nn.MSELoss()
        
        
    
    def forward(self, 
                motions:Tensor, 
                text:List[str],
                lengths:Tensor,
                tween_mask:Optional[Tensor]=None,
                padding_mask:Optional[Tensor]=None)->Tensor:
        """
        Args:
            X (Tensor): [B, L, E=263]
            text_cond (List[str]): list of string with B
            tween_mask (Tensor): [B, L, L]
            padding_mask (Tensor): padding mask

        Returns:
            Tensor: [B, L, E=263]
        """
        if isinstance(lengths, list):
            lengths = torch.Tensor(lengths)
        lengths = lengths.to(self.device)
        
        
        if tween_mask is None:
            tween_mask = self.get_tween_mask(motions=motions, lengths=lengths).to(self.device)
        if padding_mask is None:
            #try:
            padding_mask = torch.arange(motions.shape[1], device=self.device).unsqueeze(0)\
                            >= lengths.unsqueeze(1)
            padding_mask = padding_mask.to(self.device)
            # except:
            #     print(torch.arange(motions.shape[1], device=self.device).device,
            #           lengths.device)

        # This code is used to check whether successful masked by zero the X
        X_ = motions.clone().to(self.device)
        X_[tween_mask] = 0
        X_[padding_mask] = 0
        # 
        #print(X[0])
        #print(tween_mask[0,0,:])

        X_ = self.MotionInProj(X_)
        text_emb = self.text_encoder(text)
        text_emb = self.TextProj(text_emb)
        if len(text_emb.shape) < 3:
            text_emb = text_emb.unsqueeze(1)

        X_ = self.PE(X_)
        output = self.TRMEncoder.forward(
            source=X_,
            condition=text_emb,
            source_mask= padding_mask | tween_mask,
        )
        
        torch.set_printoptions(profile='full')
        result = self.MotionOutProj(output)
        return result[:,text_emb.shape[1]:,:]
    
    def compute_loss(self, 
                     motions:Tensor, 
                     text:List[str],
                     lengths:Tensor,
                     tween_mask:Optional[Tensor]=None,
                     padding_mask:Optional[Tensor]=None)->Tensor:
        motions_rec = self.forward(
            motions=motions, text=text, lengths=lengths,
            tween_mask=tween_mask, padding_mask=padding_mask
        )
        
        return self.loss_fn(motions_rec, motions)
    
    def get_tween_mask(self,
                       motions: Tensor,
                       lengths: Tensor) -> Tensor:
        """ get tween mask.
        Args:
            motions (Tensor): [B,L,E]
            lengths (Tensor): 

        Returns:
            Tensor: [B, L]
        """
        B, L,_ = motions.shape
        if self.mask_method == 'random':
            mask = torch.rand((B, L)).le(self.IPMAE.MASK.ratio)
        elif self.mask_method == 'uniform':
            mask = torch.ones((B,L))
            mask[:, ::int(1/(1-self.mask_ratio))] = 0
        return mask.bool()
    
    def save_model(self):
        torch.save(self, 'checkpoints/'+self.cfg.EXNAME+'_mae.pth')
    
    
    ########################
    
    # def get_tween_mask(self, X:Tensor, lengths:Tensor):
    #     """ Generate mask for tween
    #     Args:
    #         X (Tensor): [B,L,E=263]
    #         lengths (Tensor): [B,]

    #     Returns:
    #         Tensor: bool [B*nhead, L, L]
    #     """
    #     B, L, E = X.shape
    #     if self.mask_method == 'random':
    #         mask_idx = torch.rand((B, L)).le(self.IPMAE.MASK.ratio)

    #     elif self.mask_method == 'uniform':
    #         mask_idx = torch.ones((B, L))
    #         mask_idx[:,::int(1/(1-self.mask_ratio))] = 0
    #         #int(L*(1-self.mask_ratio))
    #         mask_idx = mask_idx.bool()
    #     else:
    #         raise NotImplementedError
        
    #     tween_mask = torch.zeros((B, L, L))
    #     tween_mask[mask_idx] = 1
    #     tween_mask = tween_mask.permute((0,2,1)) # mask for each column
    #     tween_mask = tween_mask.repeat(self.nhead,1, 1, 1)
    #     tween_mask = tween_mask.permute(1,0,2,3).reshape((-1,L,L))
    #     # a = tween_mask.bool().to(X.device)
    #     # print(a[0])
        
    #     return tween_mask.bool().to(X.device)
        