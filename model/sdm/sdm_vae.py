import torch

from torch import nn, Tensor
from model.sdm.embeddings import PositionalEncoding

class KFUVAE(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = cfg.stage
        self.keyframe_freq = cfg.keyframe_freq
        self.keyframe_only = cfg.keyframe_only
        self.d_input = cfg.d_input
        self.latent_size = cfg.latent_size
        
        self.max_kf_len = int(cfg.max_motion_len * self.keyframe_freq)+1 #20
        self.max_motion_len = cfg.max_motion_len
        
        self.device = cfg.VAE.device
        
        self.d_model = cfg.VAE.TRM.d_model
        self.nhead = cfg.VAE.TRM.nhead
        self.d_ffn = cfg.VAE.TRM.d_ffn
        self.dropout = cfg.VAE.TRM.dropout
        self.activation = cfg.VAE.TRM.activation
        self.batch_first = cfg.VAE.TRM.batch_first
                
        self.arch = cfg.VAE.arch
        
        self.encoderlayer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_ffn,
            dropout=self.dropout,
            device=self.device,
            batch_first=self.batch_first,
            activation=self.activation
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoderlayer,
            num_layers=cfg.VAE.TRM.enclayer
        )
        
        if self.arch == 'enc_dec':
            self.decoderlayer = nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.d_ffn,
                dropout=self.dropout,
                device=self.device,
                batch_first=self.batch_first,
                activation=self.activation
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=self.decoderlayer,
                num_layers=cfg.VAE.TRM.declayer
            )
        elif self.arch == 'enc_enc':
            self.decoder = nn.TransformerEncoder(
                encoder_layer=self.encoderlayer,
                num_layers=cfg.VAE.TRM.declayer
            )
            
        # motion project
        self.input_proj = nn.Linear(self.d_input, self.d_model).to(self.device)
        self.out_proj = nn.Linear(self.d_model, self.d_input).to(self.device)
        
        # latent_space for mu and logvar
        self.latent_token = nn.Parameter(
            torch.randn(self.latent_size*2, self.d_model)).to(self.device)
        
        if self.keyframe_only:
            self.encoder_pe = PositionalEncoding(d_model=self.d_model, 
                                                max_seq_len=self.max_kf_len + 2*self.latent_size)
            
            self.decoder_pe = PositionalEncoding(d_model=self.d_model, 
                                                max_seq_len=self.max_kf_len + 2*self.latent_size)
        else:
            self.encoder_pe = PositionalEncoding(d_model=self.d_model, 
                                                max_seq_len=self.max_motion_len + 2*self.latent_size)
            
            self.decoder_pe = PositionalEncoding(d_model=self.d_model, 
                                                max_seq_len=self.max_motion_len + 2*self.latent_size)
        
        # encoder decoder norm
        
        self.MSELoss_fn = nn.MSELoss()
        self.MSELoss_fn = nn.SmoothL1Loss()
        self.beta_kl = cfg.VAE.beta_kl
        
    def forward(self, motions:Tensor, lengths:Tensor):
        raise NotImplementedError
        if self.keyframe_only:
            keyframes, kf_padding_mask = self.get_keyframes(motions, lengths)
            latent_z, mu, logvar = self.encode(keyframes, kf_padding_mask)
            keyframes_rec = self.decode(latent_z, kf_padding_mask)
            return keyframes_rec, keyframes, latent_z, mu, logvar
        else:
            motion_padding =  self.get_kf_padding_mask(lengths)
            latent_z, mu, logvar = self.encode(motions, motion_padding)
            motion_rec = self.decode(latent_z, motion_padding)
            return motion_rec, motions, latent_z, mu, logvar
        

    def kl_div(self, mu:Tensor, logvar:Tensor):
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_loss
    
    def compute_loss(self, motions: Tensor, lengths: Tensor):
        keyframes_rec, keyframes, _, mu, logvar = self.forward(
            motions=motions, lengths=lengths
        )
        rec_loss = self.MSELoss_fn(keyframes, keyframes_rec)
        kl_loss = self.kl_div(mu=mu, logvar=logvar)
        
        loss = rec_loss + kl_loss * self.beta_kl
        return loss, rec_loss, kl_loss
        

    def encode(self, motions:Tensor, lengths:Tensor):
        B, _, _ = motions.shape
        
        clips, kf_padding_mask = self.get_keyframes(motions, lengths)        
        clips = self.input_proj(clips)
        latent_token = self.latent_token.repeat((B,1,1))
        latent_mask = torch.zeros((B, latent_token.shape[1]),
                                 dtype=bool,
                                 device=self.device)
        full_mask = torch.cat([latent_mask, kf_padding_mask], 1)
        # input embedding concat
        full_embedding = torch.cat([latent_token, clips],1)
        full_embedding = self.encoder_pe(full_embedding)
        full_embedding = self.encoder.forward(
            src=full_embedding, src_key_padding_mask=full_mask)
        latent_encoded = full_embedding[:,:self.latent_size*2, :]
        mu = latent_encoded[:, :self.latent_size,:]
        logvar = latent_encoded[:,self.latent_size:,:]
        assert mu.shape == logvar.shape 
        # get latent z sample
        latent_dist = torch.distributions.Normal(mu, logvar.exp().pow(0.5))
        latent_z = latent_dist.rsample()
        return latent_z, latent_dist
    
    def decode(self, latent_z:Tensor, lengths:Tensor):
        """
        Args:
            latent_z (Tensor): _description_
            kf_padding_mask (Tensor): _description_
        """
        kf_padding_mask = self.get_kf_padding_mask(lengths)
        B, K = kf_padding_mask.shape
        latent_z_mask = torch.zeros((B, latent_z.shape[1])).bool()
        
        query = torch.zeros((B, K, self.d_model), device=self.device)
        if self.arch == 'enc_enc':
            z_query = torch.cat([latent_z, query], 1)
            z_query = self.decoder_pe(z_query)
            full_mask = torch.cat([latent_z_mask, kf_padding_mask], 1)
            output = self.decoder.forward(
                src=z_query, src_key_padding_mask=full_mask
            )[B, latent_z.shape[1]:, :]
            
            
        elif self.arch == 'enc_dec':
            query = self.decoder_pe(query)
            # print(kf_padding_mask.device, self.device)
            output = self.decoder.forward(
                tgt=query, memory=latent_z, 
                tgt_key_padding_mask=kf_padding_mask
            )
        else:
            raise NotImplementedError
        
        output = self.out_proj.forward(output)
        output[kf_padding_mask] = 0
        return output
            
    
    def get_kf_padding_mask(self, lengths: Tensor):
        if not isinstance(lengths, torch.Tensor):
            lengths = torch.Tensor(lengths)
            
        if self.keyframe_only:
            kf_lengths = (lengths * self.keyframe_freq).long().to(self.device) + 1
            kf_padding_mask = torch.arange(0, self.max_kf_len).to(self.device)
            kf_padding_mask = kf_padding_mask.repeat(kf_lengths.shape[0],1)
            kf_padding_mask = (kf_padding_mask >= kf_lengths.unsqueeze(1))
        else:
            kf_lengths = torch.Tensor(lengths)
            kf_padding_mask = torch.arange(0, self.max_motion_len).to(lengths.device)
            kf_padding_mask = kf_padding_mask.repeat(len(kf_lengths),1)
            kf_padding_mask = (kf_padding_mask >= kf_lengths.unsqueeze(1))
        return kf_padding_mask.bool()

    def get_keyframes(self, motions:Tensor, lengths:Tensor):
        """ Get the keyframes from full sequences

        Args:
            X (Tensor): [B, L, E]
            lengths (Tensor): [B] long
        Returns:
            (Tensor): [B, L*maskratio,E]
            (Tensor)
        """
        if not isinstance(lengths, torch.Tensor):
            lengths = torch.Tensor(lengths).to(motions.device)

        if not self.keyframe_only:
            kf_padding_mask = self.get_kf_padding_mask(lengths=lengths).to(motions.device)
            return motions, kf_padding_mask
        keyframes = motions[:,::int(1/self.keyframe_freq)]
        assert keyframes.shape[1] == int(motions.shape[1]*self.keyframe_freq+1)
        
        kf_padding_mask = self.get_kf_padding_mask(lengths=lengths).to(motions.device)
        return keyframes, kf_padding_mask
    
    def pad_keyframes(self, keyframes: Tensor):
        if not self.keyframe_only:
            return keyframes
        B = keyframes.shape[0]
        result = torch.zeros((B, 196, self.d_input),
                             device=keyframes.device)
        result[:,::int(1/self.keyframe_freq),:] = keyframes
        return result.to(self.device)