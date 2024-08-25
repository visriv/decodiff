
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import wandb
import numpy as np

from network import *




def linear_beta_schedule(timesteps):
    if timesteps < 10:
        raise ValueError("Warning: Less than 10 timesteps require adjustments to this schedule!")

    beta_start = 0.0001 * (500/timesteps) # adjust reference values determined for 500 steps
    beta_end = 0.02 * (500/timesteps)
    betas = torch.linspace(beta_start, beta_end, timesteps)
    return torch.clip(betas, 0.0001, 0.9999)


class DiffusionModel(nn.Module):
    def __init__(self, diffusionSteps:int, condChannels:int, dataChannels:int):
        super(DiffusionModel, self).__init__()

        self.timesteps = diffusionSteps
        betas = linear_beta_schedule(timesteps=self.timesteps)

        betas = betas.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        alphas = 1.0 - betas
        alphasCumprod = torch.cumprod(alphas, axis=0)
        alphasCumprodPrev = F.pad(alphasCumprod[:-1], (0,0,0,0,0,0,1,0), value=1.0)
        sqrtRecipAlphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrtAlphasCumprod = torch.sqrt(alphasCumprod)
        sqrtOneMinusAlphasCumprod = torch.sqrt(1. - alphasCumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posteriorVariance = betas * (1. - alphasCumprodPrev) / (1. - alphasCumprod)
        sqrtPosteriorVariance = torch.sqrt(posteriorVariance)

        self.register_buffer("betas", betas)
        self.register_buffer("sqrtRecipAlphas", sqrtRecipAlphas)
        self.register_buffer("sqrtAlphasCumprod", sqrtAlphasCumprod)
        self.register_buffer("sqrtOneMinusAlphasCumprod", sqrtOneMinusAlphasCumprod)
        self.register_buffer("sqrtPosteriorVariance", sqrtPosteriorVariance)

        dim = 128
        time_dim = 128 * 2

        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        

        self.weight_network = nn.Sequential(
            nn.Linear(time_dim, int(time_dim/2)),
            nn.ReLU(),
            nn.Linear(int(time_dim/2), 1)
            # nn.Softmax(dim=-1)  # Ensure weights sum to 1
        )
        
        self.weight_network.apply(self.initialize_weights)


        # backbone model
        self.unet1 = Unet(
            dim=128,
            channels= condChannels + dataChannels,
            dim_mults=(1,1,1),
            use_convnext=True,
            convnext_mult=1,
        )

        # second branch 
        self.unet2 = Unet(
            dim=128,
            channels= condChannels + dataChannels,
            dim_mults=(1,1),
            use_convnext=True,
            convnext_mult=1,
        )

    def initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.constant_(layer.weight, 0)  # Initialize weights to zero
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)  # Initialize biases to zero


    # input shape (both inputs): B S C W H (D) -> output shape (both outputs): B S nC W H (D)
    def forward(self, conditioning:torch.Tensor, data:torch.Tensor) -> torch.Tensor:
        device = "cuda" if data.is_cuda else "cpu"
        seqLen = data.shape[1]

        # combine batch and sequence dimension for decoder processing
        d = torch.reshape(data, (-1, data.shape[2], data.shape[3], data.shape[4]))
        cond = torch.reshape(conditioning, (-1, conditioning.shape[2], conditioning.shape[3], conditioning.shape[4]))

        # TRAINING
        if self.training:

            # forward diffusion process that adds noise to data
            d = torch.concat((cond, d), dim=1)
            noise = torch.randn_like(d, device=device)
            t = torch.randint(0, self.timesteps, (d.shape[0],), device=device).long()
            dNoisy = self.sqrtAlphasCumprod[t] * d + self.sqrtOneMinusAlphasCumprod[t] * noise

            # noise prediction with network
            t_emb = self.time_mlp(t)
            weights = self.weight_network(t_emb)
            wandb.log({'average weight[0]': torch.mean(weights[:,0], dim=0).item(),
                       'min weight[0]': torch.min(weights[:,0]).item(),
                       'max weight[0]': torch.max(weights[:,0]).item(),
                       })

            # Use the unet models and delete the outputs as soon as they are no longer needed
            unet1_output = checkpoint(self.unet1, dNoisy, t)
            unet2_output = checkpoint(self.unet2, dNoisy, t)

            # predictedNoise = (
            #     weights[:].unsqueeze(1).unsqueeze(2).unsqueeze(3) * unet1_output + 
            #     weights[:].unsqueeze(1).unsqueeze(2).unsqueeze(3) * unet2_output
            # )

            # predictedNoise = self.unet1(dNoisy, t)
            predictedNoise = (
                unet1_output +
                weights[:].unsqueeze(1).unsqueeze(2) * unet2_output * 0.01
            )

            # Delete tensors if they are no longer needed
            del unet1_output, unet2_output
            torch.cuda.empty_cache()  # Clear the cache if you're on a GPU

            # unstack batch and sequence dimension again
            noise = torch.reshape(noise, (-1, seqLen, conditioning.shape[2] + data.shape[2], data.shape[3], data.shape[4]))
            predictedNoise = torch.reshape(predictedNoise, (-1, seqLen, conditioning.shape[2] + data.shape[2], data.shape[3], data.shape[4]))

            return noise, predictedNoise


        # INFERENCE
        else:
            # conditioned reverse diffusion process
            dNoise = torch.randn_like(d, device=device)
            cNoise = torch.randn_like(cond, device=device)

            for i in reversed(range(0, self.timesteps)):
                t = i * torch.ones(cond.shape[0], device=device).long()

                # compute conditioned part with normal forward diffusion
                condNoisy = self.sqrtAlphasCumprod[t] * cond + self.sqrtOneMinusAlphasCumprod[t] * cNoise

                dNoiseCond = torch.concat((condNoisy, dNoise), dim=1)

                # backward diffusion process that removes noise to create data
                # predictedNoiseCond = self.unet(dNoiseCond, t)

                # multi scale DM
                predictedNoiseCond = weights[:, 0].unsqueeze(1) * self.unet1(dNoiseCond, t) + weights[:, 1].unsqueeze(1) * self.unet1(dNoiseCond, t)

                # use model (noise predictor) to predict mean
                modelMean = self.sqrtRecipAlphas[t] * (dNoiseCond - self.betas[t] * predictedNoiseCond / self.sqrtOneMinusAlphasCumprod[t])

                dNoise = modelMean[:, cond.shape[1]:modelMean.shape[1]] # discard prediction of conditioning
                if i != 0:
                    # sample randomly (only for non-final prediction), use mean directly for final prediction
                    dNoise = dNoise + self.sqrtPosteriorVariance[t] * torch.randn_like(dNoise)

            # unstack batch and sequence dimension again
            dNoise = torch.reshape(dNoise, (-1, seqLen, data.shape[2], data.shape[3], data.shape[4]))

            return dNoise