import torch
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
import os

from transformers import RobertaModel, RobertaTokenizer

class QuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, mu=0.99):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        self.reset_codebook()
        
    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim).cuda())

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else :
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True
        
    @torch.no_grad()
    def compute_perplexity(self, code_idx) : 
        # code_idx: [B,L, ] int
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
    
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # nb_code, w
        code_count = code_onehot.sum(dim=-1)  # nb_code

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum  # w, nb_code
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count  # nb_code

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)

        self.codebook = usage * code_update + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

            
        return perplexity

    def preprocess(self, x):
        # NTC -> [NT, C]
        # x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x

    def quantize(self, x):
        # Calculate latent code x_l
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0,
                                                                                            keepdim=True)  # (N * L, b)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    
    def forward(self, x):
        '''x [N,L,C]'''
        N, T, _ = x.shape

        # Preprocess
        x = self.preprocess(x) #[N*T, C]

        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x)

        # quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else : 
            perplexity = self.compute_perplexity(code_idx)
        
        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).contiguous()   #(N, T ,DIM)
        
        return x_d, commit_loss, perplexity


class Quantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super(Quantizer, self).__init__()

        self.e_dim = e_dim
        self.n_e = n_e
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        # z in shape of [N, T, width]
        N, T, width = z.shape
        z = self.preprocess(z)
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)

        # B x V
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q - z.detach())**2) + self.beta * \
               torch.mean((z_q.detach() - z)**2)

        # preserve gradients
        z_q = z + (z_q - z).detach()
        z_q = z_q.view(N, T, -1).contiguous()   #(N, T, DIM)

        min_encodings = F.one_hot(min_encoding_indices, self.n_e).type(z.dtype)
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean*torch.log(e_mean + 1e-10)))
        return z_q, loss, perplexity

    def quantize(self, z):
        # 
        assert z.shape[-1] == self.e_dim

        # B x V
        d = torch.sum(z ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        return min_encoding_indices

    def dequantize(self, indices):

        index_flattened = indices.view(-1)
        z_q = self.embedding(index_flattened)
        z_q = z_q.view(indices.shape + (self.e_dim, )).contiguous()
        return z_q

    def preprocess(self, x):
        # NTC -> [NT, C]
        #x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x




class KFVQVAE(nn.Module):
    """
    Keyframe VQ-VAE
    """
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg 
        self.arch = cfg.KFQVAE.architecture
        self.code_dim = cfg.KFQVAE.code_dim
        self.nb_code = cfg.KFQVAE.nb_code
        self.mu = cfg.KFQVAE.mu
        self.beta = cfg.KFQVAE.beta
        
        self.device = cfg.KFQVAE.device
        # self.quantizer = QuantizeEMAReset(
        #     nb_code = self.nb_code,
        #     code_dim = self.code_dim,
        #     mu = self.mu
        # )
        self.quantizer = Quantizer(
            n_e=self.nb_code, 
            e_dim=self.code_dim,
            beta=self.beta
        ).to(self.device)
        
        
        self.max_motion_len = cfg.KFQVAE.max_motion_len
        
        self.keyframe_only = cfg.KFQVAE.keyframe_only
        self.keyframe_freq = cfg.KFQVAE.keyframe_freq
        
        self.d_model = cfg.KFQVAE.TRM.d_model   
        self.d_input = cfg.KFQVAE.d_input
        
        self.QuantizeProj = nn.Linear(self.d_model, self.code_dim).to(self.device)
        self.DequanProj = nn.Linear(self.code_dim, self.d_model).to(self.device)
        
        self.InputProj = nn.Linear(self.d_input, self.d_model).to(self.device)
        self.OutProj = nn.Linear(self.d_model, self.d_input).to(self.device)
        
        if self.arch == 'trans_enc':
            self.EncoderLayer = nn.TransformerEncoderLayer(
                d_model=cfg.KFQVAE.TRM.d_model,
                nhead=cfg.KFQVAE.TRM.nhead,
                dim_feedforward=cfg.KFQVAE.TRM.d_ffn,
                batch_first=cfg.KFQVAE.TRM.batch_first,
                device=cfg.KFQVAE.device,
                dropout=cfg.KFQVAE.TRM.dropout                
            )
            self.Encoder = nn.TransformerEncoder(self.EncoderLayer,
                                                 num_layers=cfg.KFQVAE.TRM.nlayer)
            self.Decoder = nn.TransformerEncoder(self.EncoderLayer,
                                                 num_layers=cfg.KFQVAE.TRM.nlayer)
            
            
        elif self.arch == 'trans_dec':
            self.DecoderLayer = nn.TransformerDecoderLayer(
                d_model=cfg.KFQVAE.TRM.d_model,
                nhead=cfg.KFQVAE.TRM.nhead,
                dim_feedforward=cfg.KFQVAE.TRM.d_ffn,
                batch_first=cfg.KFQVAE.TRM.batch_first,
                device=cfg.KFQVAE.device,
                dropout=cfg.KFQVAE.TRM.dropout                
            )
            self.Encoder = nn.TransformerDecoder(self.DecoderLayer,
                                                 num_layers=cfg.KFQVAE.TRM.nlayer)
            self.Decoder = nn.TransformerDecoder(self.DecoderLayer,
                                                 num_layers=cfg.KFQVAE.TRM.nlayer)
            
        elif self.arch == 'enc_dec':
            self.EncoderLayer = nn.TransformerEncoderLayer(
                d_model=cfg.KFQVAE.TRM.d_model,
                nhead=cfg.KFQVAE.TRM.nhead,
                dim_feedforward=cfg.KFQVAE.TRM.d_ffn,
                batch_first=cfg.KFQVAE.TRM.batch_first,
                device=cfg.KFQVAE.device,
                dropout=cfg.KFQVAE.TRM.dropout                
            )
            self.DecoderLayer = nn.TransformerDecoderLayer(
                d_model=cfg.KFQVAE.TRM.d_model,
                nhead=cfg.KFQVAE.TRM.nhead,
                dim_feedforward=cfg.KFQVAE.TRM.d_ffn,
                batch_first=cfg.KFQVAE.TRM.batch_first,
                device=cfg.KFQVAE.device,
                dropout=cfg.KFQVAE.TRM.dropout                
            )
            self.Encoder = nn.TransformerEncoder(self.EncoderLayer,
                                                 num_layers=cfg.KFQVAE.TRM.nlayer)
            self.Decoder = nn.TransformerDecoder(self.DecoderLayer,
                                                 num_layers=cfg.KFQVAE.TRM.nlayer)
        elif self.arch == 'mlp':
            self.Encoder = nn.Sequential(
                nn.Linear(cfg.KFQVAE.TRM.d_model, cfg.KFQVAE.TRM.d_model),
                nn.ReLU(),
                nn.Linear(cfg.KFQVAE.TRM.d_model, cfg.KFQVAE.TRM.d_model),
                nn.ReLU(),
                nn.Linear(cfg.KFQVAE.TRM.d_model, cfg.KFQVAE.TRM.d_model)
            ).to(self.device)
            
            self.Decoder = nn.Sequential(
                nn.Linear(cfg.KFQVAE.TRM.d_model, cfg.KFQVAE.TRM.d_model),
                nn.ReLU(),
                nn.Linear(cfg.KFQVAE.TRM.d_model, cfg.KFQVAE.TRM.d_model),
                nn.ReLU(),
                nn.Linear(cfg.KFQVAE.TRM.d_model, cfg.KFQVAE.TRM.d_model)
            ).to(self.device)
        else:
            raise NotImplementedError
        
        # For Text Encoder
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
            
        self.TextProj = nn.Linear(cfg.TEXTENCODER.d_model, self.d_model).to(self.device)

        self.MSEloss = nn.MSELoss()
        self.lambda_commit = cfg.KFQVAE.lambda_commit
        
        self.PE = PositionalEncoding(d_model=self.d_model, max_seq_len=200)
        
        
    def encode(self, x):
        # x [B, L, 263]
        N, T, _ = x.shape
        x_in = self.InputProj(x)
        x_encoder = self.Encoder(x_in)
        x_encoder = self.QuantizeProj(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx #[N,T]


    def forward(self, x):
        '''x in the shape of [B, 20, 263] return same shape'''
        x_in = self.PE(self.InputProj(x))
        
        # Encode
        x_encoder = self.QuantizeProj(self.Encoder(x_in))
        
        ## quantization
        x_quantized, loss, perplexity  = self.quantizer(x_encoder)

        ## decoder
        x_decoder = self.DequanProj(self.Decoder(self.PE(x_quantized)))
        x_out = self.OutProj(x_decoder)
        return x_out, loss, perplexity
    
    def forward_decoder(self, x:Tensor):
        """"
        Args:
            x (Tensor): [B, L]
        Returns:
            Tensor: [B, L, 263]
        """
        x_d = self.quantizer.dequantize(x)
        x_d = x_d.view(1, -1, self.code_dim).contiguous()
        
        # decoder
        x_decoder = self.Decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out
    
    def get_keyframes(self, motions: Tensor, offset:int=0):
        
        assert offset >=0 and offset < self.max_motion_len % int(1/self.keyframe_freq)
        # make sure offset between 0, 5
        if self.keyframe_only:            
            return motions[:,offset::int(1/self.keyframe_freq),:] # [B, 20, 263]
        tokens = torch.zeros_like(motions, device=self.device)
        tokens[:,offset::int(1/self.keyframe_freq),:] = \
            motions[:,offset::int(1/self.keyframe_freq),:]
        return tokens
    
    def compute_loss(self, keyframes:Tensor):
        """
        Args:
            keyframes (Tensor): [B, 20, 263]
        """
        keyframes_pred,commit_loss,ppl = self.forward(keyframes)
        rec_loss = self.MSEloss(keyframes, keyframes_pred)
        loss = rec_loss + self.lambda_commit * commit_loss
        return loss, rec_loss, self.lambda_commit * commit_loss



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