import torch

from torch import nn, Tensor 
from model.sdm.embeddings import PositionalEncoding, TimestepEmbedding, Timesteps
from model.sdm.operator.cross_attention import SkipTransformerEncoder, SkipTransformerDecoder
from model.sdm.operator.cross_attention import TransformerEncoderLayer, TransformerDecoderLayer
from model.chmg.ctrm import CondResTransformerEncoder
from model.chmg.ctrm import ParallelTransformer
import torch.nn.functional as F


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class KFSDMDenoiser(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.keyframe_freq = cfg.keyframe_freq
        self.latent_dim = cfg.VAE.TRM.d_model
        self.d_model = cfg.Denoiser.TRM.d_model
        self.nhead = cfg.Denoiser.TRM.nhead
        self.arch = cfg.Denoiser.arch
        self.device = cfg.Denoiser.device
        self.max_motion_len = cfg.max_motion_len #196
        self.max_kf_len = int(cfg.keyframe_freq * self.max_motion_len) + 1
        
        self.skip = cfg.Denoiser.skip
        
        # Text projection
        self.text_proj = nn.Sequential(nn.ReLU(),
                        nn.Linear(cfg.SDMTEXTENCODER.d_model, self.latent_dim)).to(self.device)
        # self.pooler_proj = nn.Sequential(nn.ReLU(),
        #                 nn.Linear(cfg.TEXTENCODER.d_model, self.latent_dim))
        
        # Time encoding
        self.tgt_pe = PositionalEncoding(d_model=self.d_model,
                                     max_seq_len=self.max_motion_len)
        
        self.mem_pe = PositionalEncoding(d_model=self.d_model,
                                     max_seq_len=self.max_motion_len)
        
        
         
        self.time_proj = Timesteps(cfg.SDMTEXTENCODER.d_model, True,0).to(self.device)
        self.time_embedding = TimestepEmbedding(cfg.SDMTEXTENCODER.d_model,
                                                    self.latent_dim).to(self.device)
        
        self.normlayer = nn.LayerNorm(self.d_model).to(self.device)
        
        self.MLPLayer = MLP(self.d_model, self.d_model*4, self.d_model, 3).to(self.device)
        
        # Main model
        if self.skip == True:
            if self.arch == 'trans_enc':
                self.TRMDenoiserLayer = TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    dim_feedforward=cfg.Denoiser.TRM.d_ffn,
                    dropout=cfg.Denoiser.TRM.dropout,
                ).to(self.device)
                self.TRMDenoiser = SkipTransformerEncoder(
                    encoder_layer=self.TRMDenoiserLayer,
                    num_layers=cfg.Denoiser.TRM.nlayer,
                    norm=nn.LayerNorm(self.d_model)
                ).to(self.device)
                
                
            elif self.arch == 'trans_dec':
                self.TRMDenoiserLayer = TransformerDecoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    dim_feedforward=cfg.Denoiser.TRM.d_ffn,
                    dropout=cfg.Denoiser.TRM.dropout,
                ).to(self.device)
                self.TRMDenoiser = SkipTransformerDecoder(
                    decoder_layer=self.TRMDenoiserLayer,
                    num_layers=cfg.Denoiser.TRM.nlayer,
                    norm=nn.LayerNorm(self.d_model)
                ).to(self.device)
            else: 
                raise NotImplementedError
        elif self.skip == 'condres':
            # condres is only valid for transformer encoder
            self.TRMDenoiserLayer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=cfg.Denoiser.TRM.d_ffn,
                dropout=cfg.Denoiser.TRM.dropout,
                device=self.device,
                batch_first=cfg.Denoiser.TRM.batch_first
            )
            self.TRMDenoiser = CondResTransformerEncoder(
                encoder_layer=self.TRMDenoiserLayer,
                num_layers=cfg.Denoiser.TRM.nlayer
            )
            
        elif self.skip == 'parallel':
            # parallel transformer for denoiser
            self.TRMDenoiser = ParallelTransformer(
                d_model=cfg.Denoiser.ParaTRM.d_model,
                nhead=cfg.Denoiser.ParaTRM.nhead,
                d_ffn=cfg.Denoiser.ParaTRM.d_ffn,
                nselflayer=cfg.Denoiser.ParaTRM.nselflayer,
                ncrosslayer=cfg.Denoiser.ParaTRM.ncrosslayer,
                dropout=cfg.Denoiser.ParaTRM.dropout,
                batch_first=cfg.Denoiser.ParaTRM.batch_first,
                activation=cfg.Denoiser.ParaTRM.activation,
                concat=cfg.Denoiser.ParaTRM.concat,
                device=cfg.device
            )
                
        elif self.skip == 'no':
            if self.arch == 'trans_enc':
                self.TRMDenoiserLayer = nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    dim_feedforward=cfg.Denoiser.TRM.d_ffn,
                    dropout=cfg.Denoiser.TRM.dropout,
                    device=self.device,
                    batch_first=cfg.Denoiser.TRM.batch_first
                )
                self.TRMDenoiser = nn.TransformerEncoder(
                    encoder_layer=self.TRMDenoiserLayer,
                    num_layers=cfg.Denoiser.TRM.nlayer
                )
                
                
            elif self.arch == 'trans_dec':
                self.TRMDenoiserLayer = nn.TransformerDecoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    dim_feedforward=cfg.Denoiser.TRM.d_ffn,
                    dropout=cfg.Denoiser.TRM.dropout,
                    device=self.device,
                    batch_first=cfg.Denoiser.TRM.batch_first
                )
                self.TRMDenoiser = nn.TransformerDecoder(
                    decoder_layer=self.TRMDenoiserLayer,
                    num_layers=cfg.Denoiser.TRM.nlayer
                )
            else: 
                raise NotImplementedError
        
        else:
            raise NotImplementedError
        
        # No TextEncoder to be added.
        
    def forward(self, 
                latents: Tensor,
                timesteps: Tensor,
                text_emb: Tensor):
        """
        Args:
            latent_sample (Tensor): noised latent sample
            [B,latent_size=1/2/3/,latent_dim=256]
            timestep (Tensor): long [B]
            text_last_hidden (Tensor): B, Lt, Et (768)
            lengths (Tensor): long [B]
        Returns:
            noise or latent [B, 4, 256]
        """
        #latents = self.input_proj(latents)
        assert self.d_model == self.latent_dim # important
        
        # time steps embedding
        # for reverse part, only one int is input
        if len(timesteps) != latents.shape[0]:
            timesteps = timesteps.expand(latents.shape[0]).clone()
            
        time_emb = self.time_proj(timesteps)
        time_emb = time_emb.to(dtype=latents.dtype)
        # [bs,1, latent_dim] <- [bs, latent_dim]
        time_emb = self.time_embedding(time_emb).unsqueeze(1)
        
        # text condition
        text_emb = self.text_proj(text_emb)
        if len(text_emb.shape) == 2:
            text_emb = text_emb.unsqueeze(1) # if pooler, fix this
        condition_emb = torch.cat([time_emb, text_emb],1)
        
        
        if self.skip == True:
            if self.arch == 'trans_enc':
                src = torch.cat([latents, condition_emb], 1)
                src = self.tgt_pe(src)
                src = src.permute(1,0,2)
                output = self.TRMDenoiser(src)[:latents.shape[1]]
                output = output.permute(1,0,2)
            
            elif self.arch == 'trans_dec':
                latents = self.tgt_pe(latents)
                condition_emb = self.mem_pe(condition_emb)
                latents = latents.permute(1,0,2)
                condition_emb = condition_emb.permute(1,0,2)
                output = self.TRMDenoiser.forward(
                    tgt=latents, memory=condition_emb
                )
                output = output.permute(1,0,2)
        elif self.skip == 'condres':
            src = self.tgt_pe(latents)
            condition_emb = self.mem_pe(condition_emb)
            output = self.TRMDenoiser.forward(
                source=src,
                condition=condition_emb,
            )[:,:latents.shape[1],:]
            output = self.MLPLayer(output)
            
        elif self.skip == 'parallel':
            latents = self.tgt_pe(latents)
            condition_emb = self.mem_pe(condition_emb)
            output = self.TRMDenoiser.forward(
                left=latents,
                right=condition_emb
            )
        
        elif self.skip == 'no':
            if self.arch == 'trans_enc':
                src = torch.cat([latents, condition_emb], 1)
                src = self.tgt_pe(src)
                output = self.TRMDenoiser(src)[:, :latents.shape[1],:]
            
            elif self.arch == 'trans_dec':
                latents = self.tgt_pe(latents)
                condition_emb = self.mem_pe(condition_emb)
                output = self.TRMDenoiser.forward(
                    tgt=latents, memory=condition_emb
                )
        else:
            raise NotImplementedError
        
        return output