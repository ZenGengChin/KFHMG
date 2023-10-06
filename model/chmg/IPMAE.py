import torch 

import os 


from typing import List, Optional
from torch import nn, Tensor
from omegaconf import DictConfig
from transformers import RobertaModel, RobertaTokenizer
from transformers import T5Model, T5Tokenizer
from transformers import ElectraModel, ElectraTokenizer

# from model.sdm.constraints import SequenceSmoother


# global_smoother = SequenceSmoother(input_size=263, kernel_size=1).cuda()

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
        
        self.TRMDecoderLayer = nn.TransformerDecoderLayer(
            d_model=cfg.IPMAE.TRMDECODER.d_model,
            nhead=cfg.IPMAE.TRMDECODER.nhead,
            dim_feedforward=cfg.IPMAE.TRMDECODER.d_ffn,
            batch_first=cfg.IPMAE.TRMDECODER.batch_first,
            dropout=cfg.IPMAE.TRMDECODER.dropout,
            device=cfg.IPMAE.device)
        
        self.TRMDecoder = nn.TransformerDecoder(
            decoder_layer=self.TRMDecoderLayer,
            num_layers=cfg.IPMAE.TRMDECODER.nlayer
        )
        
        self.mask_method = cfg.IPMAE.MASK.method 
        self.mask_ratio = cfg.IPMAE.MASK.ratio
        self.nhead = cfg.IPMAE.TRMDECODER.nhead

        TextEncoderList =  cfg.TEXTENCODERLIST
        TextModelList = cfg.TEXTENCODERMODELLIST
        assert cfg.TEXTENCODER.NAME in TextEncoderList, NotImplementedError
        
        TextEncoderPath = cfg.TEXTENCODER.ENCODERPATH
        TokenizerPath = cfg.TEXTENCODER.ENCODERPATH
        if not os.path.exists(TokenizerPath):
            TokenizerPath = TextModelList[TextEncoderList.index(cfg.TEXTENCODER.NAME)]
        self.TextEncoder = eval(cfg.TEXTENCODER.NAME).from_pretrained(TextEncoderPath).to(self.device)
        self.Tokenizer = eval(cfg.TEXTENCODER.TOKENIZER).from_pretrained(TextEncoderPath)
        # Store the model for the first time
        if not os.path.exists(TokenizerPath):
            self.TextEncoder.save_pretrained(os.path.join('pretraining',TextEncoderPath))
            self.Tokenizer.save_pretrained(os.path.join('pretraining',TextEncoderPath))
            
        self.TextProj = nn.Linear(cfg.TEXTENCODER.d_model, cfg.IPMAE.TRMDECODER.d_model).to(self.device)
        self.loss_fn = nn.MSELoss()
                
    def text_encode(self, text:List[str]):
        """
        Args:
            text (List[str]): length of [B,]
        
        Returns:
            Tensor: [B, L, E=768], bool mask F for unmask, T for mask
        """
        tokens = self.Tokenizer(text, return_tensors='pt', padding=True).to(self.device)
        outputs = self.TextEncoder(**tokens)
        return outputs, ~(tokens.attention_mask.bool())
    
    def forward(self, 
                X:Tensor, 
                text_cond:List[str],
                lengths:Tensor,
                tween_mask:Optional[Tensor]=None,
                padding_mask:Optional[Tensor]=None)->Tensor:
        """_summary_

        Args:
            X (Tensor): [B, L, E=263]
            text_cond (List[str]): list of string with B
            tween_mask (Tensor): [B, L, L]
            padding_mask (Tensor): padding mask

        Returns:
            Tensor: [B, L, E=263]
        """

        
        if tween_mask is None:
            tween_mask = self.get_tween_mask(X=X, lengths=lengths)
        if padding_mask is None:
            padding_mask = torch.arange(X.shape[1], device=self.device).unsqueeze(0)\
                            < lengths.unsqueeze(1)

        # This code is used to check whether successful masked by zero the X
        X_zeros = tween_mask[0::self.nhead, 0, :].squeeze(-1)
        X_ = X.clone()
        #X_ += torch.randn_like(X_) * 0.05
        X_[X_zeros] = 0
        X_[~padding_mask] = 0
        # torch.set_printoptions(profile='full')
        #print(X[0])
        #print(tween_mask[0,0,:])

        X_ = self.MotionInProj(X_)
        text_emb, text_padding_mask = self.text_encode(text_cond)
        text_emb = self.TextProj(text_emb.last_hidden_state)

        X_ = self.PE(X_)
        output = self.TRMDecoder.forward(
            tgt=X_, memory=text_emb,
            tgt_mask=tween_mask, memory_mask=None,
            tgt_key_padding_mask= ~padding_mask, 
            memory_key_padding_mask= text_padding_mask
        )
        
        result = self.MotionOutProj(output)
        # result = global_smoother.forward(result)
        
        return result
    
    def compute_loss(self, 
                     X:Tensor, 
                     text_cond:List[str],
                     lengths:Tensor,
                     tween_mask:Optional[Tensor]=None,
                     padding_mask:Optional[Tensor]=None)->Tensor:
        X_rec = self.forward(
            X=X, text_cond=text_cond, lengths=lengths,
            tween_mask=tween_mask, padding_mask=padding_mask
        )
        
        return self.loss_fn(X_rec, X)
    
    def get_tween_mask(self, X:Tensor, lengths:Tensor):
        """ Generate mask for tween
        Args:
            X (Tensor): [B,L,E=263]
            lengths (Tensor): [B,]

        Returns:
            Tensor: bool [B*nhead, L, L]
        """
        B, L, E = X.shape
        if self.mask_method == 'random':
            mask_idx = torch.rand((B, L)).le(self.IPMAE.MASK.ratio)

        elif self.mask_method == 'uniform':
            mask_idx = torch.ones((B, L))
            mask_idx[:,::int(1/(1-self.mask_ratio))] = 0
            #int(L*(1-self.mask_ratio))
            mask_idx = mask_idx.bool()
        else:
            raise NotImplementedError
        
        tween_mask = torch.zeros((B, L, L))
        tween_mask[mask_idx] = 1
        tween_mask = tween_mask.permute((0,2,1)) # mask for each column
        tween_mask = tween_mask.repeat(self.nhead,1, 1, 1)
        tween_mask = tween_mask.permute(1,0,2,3).reshape((-1,L,L))
        # a = tween_mask.bool().to(X.device)
        # print(a[0])
        
        return tween_mask.bool().to(X.device)
        