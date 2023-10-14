import inspect
import numpy as np
import torch

from model.sdm.losses.kl import KLLoss

import torch 

from torch import nn
from model.sdm.sdm_denoiser import KFSDMDenoiser
from model.sdm.sdm_vae import KFUVAE
from model.sdm.sdm_textencoder import KFTextEncoder
from model.chmg.IPMAE import InterpolateMAE

from diffusers import DDPMScheduler, DDIMScheduler


class SDM(nn.Module):   
    """
    Stage 1 vae
    Stage 2 diffusion
    [Stage 3] interpolation
    """

    def __init__(self, cfg, **kwargs):
        super().__init__()

        self.cfg = cfg

        self.stage = cfg.stage
        self.nfeats = cfg.DATALOADER.NFEATS
        self.latent_dim = cfg.latent_dim
        self.latent_size = cfg.latent_size
        self.guidance_scale = cfg.guidance_scale
        self.guidance_uncodp = cfg.guidance_uncondp
        self.device = 'cuda'
        self.keyframe_only = cfg.keyframe_only
        
        
        self.vae = KFUVAE(cfg=cfg)
        
        if self.stage == "diffusion":
            self.text_encoder = KFTextEncoder(cfg=cfg)
            self.vae.training = False
            for p in self.vae.parameters():
                p.requires_grad = False

            self.denoiser = KFSDMDenoiser(cfg=cfg)
            # Noise Scheduler
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=cfg.NoiseScheduler.num_train_timesteps,
                beta_start=cfg.NoiseScheduler.beta_start,
                beta_end=cfg.NoiseScheduler.beta_end,
                beta_schedule=cfg.NoiseScheduler.beta_schedule,
                clip_sample=cfg.NoiseScheduler.clip_sample
            )
            
            self.scheduler = DDIMScheduler(
                num_train_timesteps=cfg.NoiseScheduler.num_train_timesteps,
                beta_start=cfg.NoiseScheduler.beta_start,
                beta_end=cfg.NoiseScheduler.beta_end,
                beta_schedule=cfg.NoiseScheduler.beta_schedule,
                clip_sample=cfg.NoiseScheduler.clip_sample,
                set_alpha_to_one=cfg.NoiseScheduler.set_alpha_to_one,
                steps_offset=cfg.NoiseScheduler.steps_offset
            )
            self.do_classifier_free_guidance = self.guidance_scale > 1.0
            
        if self.cfg.keyframe_only:
            print('Load Interpolation network from:', self.cfg.IPMAE.path)
            self.interpolator = InterpolateMAE(cfg=cfg)
            self.interpolator.load_state_dict(torch.load(self.cfg.IPMAE.path))
            
        
        self.vae_loss_fn = nn.SmoothL1Loss()
        self.denoise_loss_fn = nn.MSELoss()
        self.kl_loss_fn = KLLoss()
              
        # evaluator:
        

    def forward(self, motions, text, lengths):
        texts = text
        motions = motions.to(self.device)
        if not isinstance(lengths, torch.Tensor):
            lengths = torch.Tensor(lengths).to(motions.device)
            
        if self.stage in ['diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                uncond_tokens.extend(texts)
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            z = self._diffusion_reverse(text_emb, lengths)
        elif self.stage in ['vae']:            
            z, dist_m = self.vae.encode(motions, lengths)

        with torch.no_grad():
            # ToDo change mcross actor to same api
            feats_rst = self.vae.decode(z, lengths)
            
        if self.keyframe_only:
            feats_rst = self.vae.pad_keyframes(feats_rst)
            feats_rst = self.interpolator.forward(
                feats_rst,
                text,
                lengths
            )
        return feats_rst


    
    
    def _diffusion_reverse(self, encoder_hidden_states, lengths=None):
        # init latents
        bsz = encoder_hidden_states.shape[0]
        if self.do_classifier_free_guidance:
            bsz = bsz // 2

        latents = torch.randn(
            (bsz, self.latent_size, self.latent_dim),
            device=encoder_hidden_states.device,
            dtype=torch.float)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(
            self.cfg.NoiseScheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(encoder_hidden_states.device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.NoiseScheduler.eta

        # reverse
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (torch.cat(
                [latents] *
                2) if self.do_classifier_free_guidance else latents)
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            noise_pred = self.denoiser(
                latents=latent_model_input,
                timesteps=t.unsqueeze(0), # 0-d tensor to 1-d
                text_emb=encoder_hidden_states,
            )
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents,
                                              **extra_step_kwargs).prev_sample


        return latents
    

    def _diffusion_process(self, latents, encoder_hidden_states, lengths=None):
        # our latent   [batch_size, n_token=1 or 5 or 10, latent_dim=256]
        # sd  latent   [batch_size, [n_token0=64,n_token1=64], latent_dim=4]
        # [n_token, batch_size, latent_dim] -> [batch_size, n_token, latent_dim]

        # Sample noise that we'll add to the latents
        # [batch_size, n_token, latent_dim]
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz, ),
            device=latents.device,
        )
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents.clone(), noise,
                                                       timesteps)
        # Predict the noise residual
        noise_pred = self.denoiser.forward(
            latents=noisy_latents,
            timesteps=timesteps,
            text_emb=encoder_hidden_states
        )
        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.

        n_set = {
            "noise": noise,
            "noise_pred": noise_pred,
        }
        return n_set

    def train_vae_forward(self, motions, lengths):

        motion_z, dist_m = self.vae.encode(motions, lengths)
        feats_rst = self.vae.decode(motion_z, lengths)


        if dist_m is not None:
            # Create a centred normal distribution to compare with
            mu_ref = torch.zeros_like(dist_m.loc)
            scale_ref = torch.ones_like(dist_m.scale)
            dist_ref = torch.distributions.Normal(mu_ref, scale_ref)

        # cut longer part over max length
        
        if self.keyframe_only:
            motions,_ = self.vae.get_keyframes(motions, lengths)
        
        rs_set = {
            "m_ref": motions,
            "m_rst": feats_rst,
            "lat_m": motion_z,
            "dist_m": dist_m,
            "dist_ref": dist_ref,
        }
        return rs_set

    def train_diffusion_forward(self, motions, text, lengths):
        feats_ref = motions
        lengths = lengths
        # motion encode
        with torch.no_grad():
            z, dist = self.vae.encode(feats_ref, lengths)

        # classifier free guidance: randomly drop text during training
        text = [
            "" if np.random.rand(1) < self.guidance_uncodp else i
            for i in text
        ]
            # text encode
        cond_emb = self.text_encoder(text)

        # diffusion process return with noise and noise_pred
        n_set = self._diffusion_process(z, cond_emb, lengths)
        return {**n_set}

    def test_diffusion_forward(self, text, lengths):
        lengths = lengths

        # get text embeddings
        if self.do_classifier_free_guidance:
            uncond_tokens = [""] * len(lengths)
            texts = text
            uncond_tokens.extend(texts)
            texts = uncond_tokens
        cond_emb = self.text_encoder(texts)


        # diffusion reverse
        with torch.no_grad():
            z = self._diffusion_reverse(cond_emb, lengths)

        with torch.no_grad():
            feats_rst = self.vae.decode(z, lengths)
        return feats_rst

    def compute_loss(self, motions, text, lengths):
        if self.stage == 'vae':
            vae_rst = self.train_vae_forward(motions, lengths)
            rec_loss = self.vae_loss_fn(vae_rst['m_ref'], vae_rst['m_rst'])
            kl_loss = self.kl_loss_fn(vae_rst['dist_m'], vae_rst['dist_ref'])
            loss = rec_loss + kl_loss * self.cfg.VAE.beta_kl
            return loss, rec_loss, kl_loss
        elif self.stage == 'diffusion':
            n_rst = self.train_diffusion_forward(motions, text, lengths)
            noise_pred = n_rst['noise_pred']
            noise = n_rst['noise']
            loss = self.denoise_loss_fn(noise_pred, noise)
            return loss
        else:
            raise NotImplementedError
        
    def save_model(self):
        path_name = 'checkpoints/'+self.cfg.EXNAME + '_vae.pth'
        dn_path_name = 'checkpoints/'+self.cfg.EXNAME + '_dn.pth'
        if self.stage == 'vae':
            torch.save(self.vae, path_name)
            print('Save VAE model to', path_name)
        elif self.stage == 'diffusion':
            torch.save(self.denoiser, dn_path_name)
            print('Save DN model to', dn_path_name)
            
