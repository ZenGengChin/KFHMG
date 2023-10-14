import torch 
from copy import deepcopy
from torch import nn, Tensor
from typing import Optional

from torch.nn import MultiheadAttention

class CondResTransformerEncoder(nn.Module):
    def __init__(self,
                 encoder_layer: nn.TransformerEncoderLayer,
                 num_layers:int,
                 condition_plus:bool = True) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [deepcopy(encoder_layer) for i in range(num_layers)]
        )
        self.condition_plus = condition_plus
    
    def forward(self,
                source: Tensor,
                condition:Optional[Tensor]=None,
                source_mask: Optional[Tensor]=None
                ):
        """
        All the tensor are batch_first
        Args:
            source (Tensor): [B, L_s, E]
            condition (Optional[Tensor], optional): [B, L_c, E] Defaults to None.
                if None, it will be a plain transformer encoder.
                if given, condition residual will be applied. 
            source_mask (Optional[Tensor], optional): [B, L_s]. Defaults to None.
        """
        # if condition is not None:
        #     condition_mask = torch.zeros((condition.shape[0], condition.shape[1]),
        #                                  dtype=bool).to(source.device)
        #     if source_mask is None:
        #         source_mask = torch.zeros((source.shape[0], source.shape[1]),
        #                                   dtype=bool, device=source.device)
        #     mask = torch.cat([condition_mask, source_mask], dim=1)
        #     output = torch.cat([torch.zeros_like(condition).to(source.device),
        #                         source], dim=1)
        #     for layer in self.layers:
        #         if self.condition_plus:
        #             output[:, :condition.shape[1], :] += condition
        #             output = layer(src=output, src_key_padding_mask=mask)
        #         else:
        #             output[:,:condition.shape[1],:]  = condition
        #             output = layer(src=output, src_key_padding_mask=mask)
                    
        # else:
        #     output = source
        #     for layer in self.layers:
        #         output = layer.forward(src=output, src_key_padding_mask=source_mask)
        # return output
        if condition is not None:
            condition_mask = torch.zeros((condition.shape[0], condition.shape[1]),
                                         dtype=bool).to(source.device)
            if source_mask is None:
                source_mask = torch.zeros((source.shape[0], source.shape[1]),
                                          dtype=bool, device=source.device)
            mask = torch.cat([source_mask, condition_mask], dim=1)
            output = torch.cat([source, torch.zeros_like(condition).to(source.device)],
                               dim=1)
            
            for layer in self.layers:
                if self.condition_plus:
                    output[:, -condition.shape[1]:, :] += condition
                    output = layer(src=output, src_key_padding_mask=mask)
                else:
                    output[:,-condition.shape[1]:,:]  = condition
                    output = layer(src=output, src_key_padding_mask=mask)
                    
        else:
            output = source
            for layer in self.layers:
                output = layer.forward(src=output, src_key_padding_mask=source_mask)
        return output
    

class CoTransformerHalfLayer(nn.Module):
    def __init__(self,
                 d_model:int=256,
                 nhead:int=4,
                 d_ffn:int=1024,
                 device:str='cuda',
                 dropout:float=0.1,
                 activation:str='relu',
                 batch_first:bool=True) -> None:
        super().__init__()
        self.device = device
        self.d_model = d_model 
        self.nhead = nhead
        self.d_ffn = d_ffn
        self.dropout = dropout
        self.batch_first = batch_first
        
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()
        
        self.mha = MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.nhead,
            dropout=self.dropout,
            batch_first=True,
            device=self.device
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_ffn),
            self.activation,
            nn.Linear(self.d_ffn, d_model)
        ).to(self.device)
        self.normlayer1 = nn.LayerNorm(self.d_model).to(self.device)
        self.normlayer2 = nn.LayerNorm(self.d_model).to(self.device)
        
    def forward(self, query:Tensor, kv:Tensor):
        x,_ = self.mha.forward(query=query, key=kv, value=kv)
        x = self.normlayer1(query + x)
        x = self.normlayer2(x + self.ffn.forward(x))
        return x
    
class CoTransformerLayer(nn.Module):
    def __init__(self,
                 d_model:int=256,
                 nhead:int=4,
                 d_ffn:int=1024,
                 device:str='cuda',
                 dropout:float=0.1,
                 activation:str='relu',
                 batch_first:bool=True) -> None:
        super().__init__()
        self.LeftCoTRMLayer = CoTransformerHalfLayer(
            d_model=d_model,
            nhead=nhead,
            d_ffn=d_ffn,
            device=device,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first
        )
        
        self.RightCoTRMLayer = deepcopy(self.LeftCoTRMLayer)
        
        self.LeftTRMLayer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ffn,
            dropout=dropout,
            device=device,
            activation=activation,
            batch_first=batch_first
        )
        
        self.RightTRMLayer = deepcopy(self.LeftTRMLayer)
        
    def forward(self, left: Tensor, right:Tensor):
        """
        Args:
            left (Tensor): [B, Ll, E]
            right (Tensor): [B, Lr, E]
        Returns:
            left, right
        """
        left = self.LeftCoTRMLayer.forward(
            query=left,
            kv=right
        )
        left = self.LeftTRMLayer.forward(src=left)
        
        right = self.RightCoTRMLayer.forward(
            query=right,
            kv=left
        )
        
        right = self.RightTRMLayer.forward(src=right)
        
        return left, right
        

class ParallelTransformer(nn.Module):
    def __init__(self,
                 d_model: int=256,
                 nhead: int=4, 
                 d_ffn: int=1024,
                 nselflayer: int = 3,
                 ncrosslayer: int = 3,
                 dropout: float=0.1,
                 batch_first: bool=True,
                 activation: str='relu',
                 concat: str='linear',
                 device: str='cuda'
                 ) -> None:
        super().__init__()
        self.device = device
        
        self.SelfLayer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ffn,
            activation=activation,
            batch_first=batch_first,
            device=device
        )        
        self.LeftSelfLayers = nn.ModuleList([deepcopy(self.SelfLayer)
                                             for i in range(nselflayer)])
        self.RightSelfLayers = nn.ModuleList([deepcopy(self.SelfLayer)
                                             for i in range(nselflayer)])
        
        self.CrossLayer = CoTransformerLayer(
            d_model=d_model,
            nhead=nhead,
            d_ffn=d_ffn,
            device=device,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first
        )
        
        self.CrossTransformer = nn.ModuleList([deepcopy(self.CrossLayer)
                                               for i in range(ncrosslayer)])
        self.concat = concat
        
        if self.concat == 'linear':
            self.OutputLayer = nn.Linear(2*d_model, d_model).to(self.device)
        elif self.concat == 'trunc':
            self.OutputLayer = deepcopy(self.SelfLayer)
            
    def forward(self,left: Tensor, right:Tensor)->Tensor:
        for layer in self.LeftSelfLayers:
            left = layer.forward(src=left)
        
        for layer in self.RightSelfLayers:
            right = layer.forward(src=right)
        
        for layer in self.CrossTransformer:
            left, right = layer.forward(
                left = left,
                right = right
            )
        if self.concat == 'linear':
            output = torch.cat([left, right], dim=-1)
            output = self.OutputLayer(output)
        elif self.concat == 'trunc':
            output = torch.cat([left, right], dim=1)
            output = self.OutputLayer(output)[:,:left.shape[1],:]
        return output 
        