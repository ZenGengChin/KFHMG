import torch

from torch import nn, Tensor 
from model.sdm.embeddings import PositionalEncoding, TimestepEmbedding, Timesteps
from model.sdm.operator.cross_attention import SkipTransformerEncoder, SkipTransformerDecoder
from model.sdm.operator.cross_attention import TransformerEncoderLayer, TransformerDecoderLayer

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
        
        # Main model
        if self.skip:
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
        else:
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
            noise or latent
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
        
        
        if self.skip:
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
        
        else:
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
        return output