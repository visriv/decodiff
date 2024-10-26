import math
import torch
import torch.nn.functional as F

from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from functools import partial
# from Models.interpretable_diffusion.transformer import Transformer
# from Models.interpretable_diffusion.model_utils import default, identity, extract






from torch.utils.checkpoint import checkpoint
import wandb
import numpy as np

from .network import *


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# gaussian diffusion trainer class

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class DiffusionDenoise(nn.Module):
    def __init__(
            self,
            config
    ):
        super(DiffusionDenoise, self).__init__()

        self.use_ff = config.model.use_ff
        self.seq_length = config.model.seq_length
        self.feature_size = config.model.feature_size
        self.ff_weight = default(config.model.fourier_reg_weight, math.sqrt(config.model.seq_length) / 5)
        self.data_channels = config.model.data_channels
        self.cond_channels = config.model.input_steps * self.data_channels
        self.dim = config.model.dim
        self.clip_denoised = config.model.clip_denoised

        # backbone model
        self.unet1 = Unet(
            dim=self.dim,
            channels= self.cond_channels + self.data_channels,
            dim_mults=(1,1,1),
            use_convnext=True,
            convnext_mult=1,
        )

        if config.model.beta_schedule == 'linear':
            betas = linear_beta_schedule(config.model.diffusion_steps)
        elif config.model.beta_schedule == 'cosine':
            betas = cosine_beta_schedule(config.model.diffusion_steps)
        else:
            raise ValueError(f'unknown beta schedule {config.model.beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)


        self.diffusion_steps = int(config.model.diffusion_steps)
        self.loss_type = config.model.loss_type



        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas', alphas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate reweighting
        register_buffer('loss_weight', torch.sqrt(alphas) * torch.sqrt(1. - alphas_cumprod) / betas / 100)



    def forward(self, data, conditioning, **kwargs):
        

        # combine batch and sequence dimension for decoder processing
        d = torch.reshape(data, (-1, data.shape[2], data.shape[3], data.shape[4]))
        cond = torch.reshape(conditioning, (-1, conditioning.shape[2], conditioning.shape[3], conditioning.shape[4]))



        if self.training:
            b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
            assert n == feature_size, f'number of variable must be {feature_size}'
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            return self._train_loss(x_start=x, t=t, **kwargs)
        else:
            return self.sample(d.shape, cond)

    def model(self, x, t):
        return self.unet1(x, t, context=None)
    
    def _train_loss(self, x0, t, target=None, noise=None, padding_masks=None):
        noise = default(noise, lambda: torch.randn_like(x0))
        if target is None:
            target = x0

        x_t =  extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 + \
            extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise

        x0_hat = self.model(x_t, t)

        train_loss = self.loss_fn(target, x0_hat, reduction='none')

        fourier_loss = torch.tensor([0.])
        if self.use_ff:
            fft1 = torch.fft.fft(x0_hat.transpose(1, 2), norm='forward')
            fft2 = torch.fft.fft(target.transpose(1, 2), norm='forward')
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fourier_loss = self.loss_fn(torch.real(fft1), torch.real(fft2), reduction='none')\
                           + self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction='none')
            train_loss +=  self.ff_weight * fourier_loss
        
        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)
        return train_loss.mean()

    @torch.no_grad()
    def sample(self, shape, cond): 
        device = self.betas.device
        xT = torch.randn(shape, device=device)
        xt = xT

        cNoise = torch.randn_like(cond, device=device)

        for t in tqdm(reversed(range(0, self.num_timesteps)),
                    desc='sampling loop time step', total=self.diffusion_steps):
            
            condNoisy = extract(self.sqrt_alphas_cumprod, t, cond.shape) * cond + \
                extract(self.sqrt_one_minus_alphas_cumprod, t, cond.shape) * cNoise
        
            x_tminus1 = self.p_sample(xt, t, clip_denoised = self.clip_denoised, cond = condNoisy)
            xt = x_tminus1
        x0 = x_tminus1
        return x0





    def p_sample(self, xt, t: int, clip_denoised=True, cond = None):
        batched_times = torch.full((xt.shape[0],), t, device=xt.device, dtype=torch.long)
        model_mean, _, model_log_variance = \
            self.p_mean_variance(x=xt, t=batched_times, clip_denoised=clip_denoised, cond=cond)
        noise = torch.randn_like(xt) if t > 0 else 0.  # no noise if t == 0

        x_tminus1 = model_mean + (0.5 * model_log_variance).exp() * noise
        return x_tminus1


    def p_mean_variance(self, x_t, t, clip_denoised=True, cond=None):
        #conditioned noise: condNoisy | noise
        dNoiseCond = torch.concat((cond, x_t), dim=1)

        x0_hat = self.model(dNoiseCond, t, context=None)
        if clip_denoised:
            x0_hat.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior(x0=x0_hat, x_t=x_t, t=t)
        return model_mean, posterior_variance, posterior_log_variance


    def q_posterior(self, x0, x_t, t):
        # return q(x_{t-1} | xt, x0)
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x0 +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped



    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')







    def q_sample(self, x0, t, noise=None):
        # get q(x_t | x0)
        noise = default(noise, lambda: torch.randn_like(x0))
        return (
                extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )






if __name__ == '__main__':
    pass