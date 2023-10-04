import transformers
import torch 
import os

from transformers import CLIPModel, CLIPProcessor
from transformers import RobertaModel, RobertaTokenizer
from torch import nn 
from typing import List


class KFTextEncoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        TEXTENCODERList =  cfg.TEXTENCODERLIST
        TextModelList = cfg.TEXTENCODERMODELLIST
        assert cfg.SDMTEXTENCODER.NAME in TEXTENCODERList, NotImplementedError
        
        SDMTEXTENCODERPath = cfg.SDMTEXTENCODER.ENCODERPATH
        TokenizerPath = cfg.SDMTEXTENCODER.ENCODERPATH
        if not os.path.exists(TokenizerPath):
            TokenizerPath = TextModelList[TEXTENCODERList.index(cfg.SDMTEXTENCODER.NAME)]
        if not os.path.exists(SDMTEXTENCODERPath):
            SDMTEXTENCODERPath = TextModelList[TEXTENCODERList.index(cfg.SDMTEXTENCODER.NAME)]
        self.SDMTEXTENCODER = eval(cfg.SDMTEXTENCODER.NAME).from_pretrained(SDMTEXTENCODERPath).to(self.device)
        self.Tokenizer = eval(cfg.SDMTEXTENCODER.TOKENIZER).from_pretrained(SDMTEXTENCODERPath)
        # Store the model for the first time
        if not os.path.exists(TokenizerPath):
            self.SDMTEXTENCODER.save_pretrained(os.path.join('pretraining',SDMTEXTENCODERPath))
            self.Tokenizer.save_pretrained(os.path.join('pretraining',SDMTEXTENCODERPath))
            
        #self.pooler_proj = nn.Linear(cfg.SDMTEXTENCODER.d_model, self.d_model).to(self.device)
        #self.last_hidden_proj = nn.Linear(cfg.SDMTEXTENCODER.d_model, self.d_model).to(self.device)
        
        # set vae and SDMTEXTENCODER frozen.
        for para in self.SDMTEXTENCODER.parameters():
            para.requires_grad = False
            
        print('Loading frozen text encoder',cfg.SDMTEXTENCODER.NAME,'from:', SDMTEXTENCODERPath)
            
    def forward(self, text: List[str]):
        with torch.no_grad():
            if self.cfg.SDMTEXTENCODER.NAME == 'CLIPModel':
                tokens = self.Tokenizer(text, return_tensors='pt', padding=True, 
                                        truncation=True).to(self.device)
                text_encoded = self.SDMTEXTENCODER.get_text_features(**tokens)
            else:
                tokens = self.Tokenizer(text, return_tensors='pt', padding=True).to(self.device)
                text_encoded = self.SDMTEXTENCODER(**tokens)
                text_encoded = text_encoded.pooler_output
                if self.cfg.text_rep == 'pooler':
                    pass
                elif self.cfg.text_rep == 'poolerhidden':
                    pass 
                else:
                    pass         
        return text_encoded #, ~(tokens.attention_mask.bool())