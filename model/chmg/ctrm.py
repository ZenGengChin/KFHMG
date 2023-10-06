import torch 
from copy import deepcopy
from torch import nn, Tensor
from typing import Optional

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
                if None, it will be plain transformer encoder.
                if given, condition residual will be applied. 
            source_mask (Optional[Tensor], optional): [B, L_s]. Defaults to None.
        """
        if condition is not None:
            condition_mask = torch.zeros((condition.shape[0], condition.shape[1]),
                                         dtype=bool).to(source.device)
            if source_mask is None:
                source_mask = torch.zeros((source.shape[0], source.shape[1]),
                                          dtype=bool, device=source.device)
            mask = torch.cat([condition_mask, source_mask], dim=1)
            output = torch.cat([torch.zeros_like(condition).to(source.device),
                                source], dim=1)
            for layer in self.layers:
                if self.condition_plus:
                    output[:, :condition.shape[1], :] += condition
                    output = layer(src=output, src_key_padding_mask=mask)
                else:
                    output[:,:condition.shape[1],:]  = condition
                    output = layer(src=output, src_key_padding_mask=mask)
                    
        else:
            output = source
            for layer in self.layers:
                output = layer.forward(src=output, src_key_padding_mask=source_mask)
        return output