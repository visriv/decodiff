
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import wandb
import numpy as np

from .network import *




def linear_beta_schedule(timesteps):
    if timesteps < 10:
        raise ValueError("Warning: Less than 10 timesteps require adjustments to this schedule!")

    beta_start = 0.0001 * (500/timesteps) # adjust reference values determined for 500 steps
    beta_end = 0.02 * (500/timesteps)
    betas = torch.linspace(beta_start, beta_end, timesteps)
    return torch.clip(betas, 0.0001, 0.9999)


class DiffusionModel(nn.Module):
    def __init__(self, 
                config):
        super(DiffusionModel, self).__init__()
        self.config = config
        self.timesteps = config.model.diffusion_steps
        betas = linear_beta_schedule(timesteps=self.timesteps)
        self.data_channels = config.model.data_channels
        self.cond_channels = config.model.input_steps * self.data_channels
        self.dim = config.model.dim

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




        # backbone model
        self.unet1 = Unet(
            dim=self.dim,
            channels= self.cond_channels + self.data_channels,
            dim_mults=(1,1,1),
            use_convnext=True,
            convnext_mult=1,
        )

        # second branch 
        if (config.model.twin_tower):
            self.unet2 = Unet(
                dim=self.dim,
                channels= self.cond_channels + self.data_channels,
                dim_mults=(1,1), #self.config.model.unet2_mults,
                use_convnext=True,
                convnext_mult=1,
            )
            
            if (config.model.fusion):
                self.FusionModule = FusionModule(config)
        # Load weights for net1 if a path is provided
        # if net1_weights_path is not None:
        #     self.net1.load_state_dict(torch.load(net1_weights_path))
        #     print("Loaded net1 weights from:", net1_weights_path)


        # # Freeze the parameters of net1
        # for param in self.net1.parameters():
        #     param.requires_grad = False




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
 




            if (( self.config.model.twin_tower == True) and ( self.config.model.control_connect == True)):
                unet2_output, intermediate_outputs2 = self.unet2(dNoisy, t, context=None)
                unet1_output, intermediate_outputs1 = self.unet1(dNoisy, t, context=intermediate_outputs2['upsample_20'])
                predictedNoise = unet1_output
                del unet1_output, unet2_output

            elif (( self.config.model.twin_tower == True) and ( self.config.model.fusion == True)):
                unet2_output, intermediate_outputs2 = self.unet2(dNoisy, t, context=None)
                unet1_output, intermediate_outputs1 = self.unet1(dNoisy, t, context=None)
                predictedNoise = self.FusionModule(unet1_output, unet2_output, t)
                del unet1_output, unet2_output

            elif ( self.config.model.twin_tower == False):
                unet1_output, intermediate_outputs1 = self.unet1(dNoisy, t, context=None)
                predictedNoise = unet1_output
                del unet1_output

            
            # Delete tensors if they are no longer needed
            torch.cuda.empty_cache()  # Clear the cache if you're on a GPU

            # once denoising is completing, save interm outputs of the UNet
            interm_features = {'UNet1': intermediate_outputs1,
                               'UNet2': intermediate_outputs2 if self.config.model.twin_tower == True else {}
                    }
            
        
            # unstack batch and sequence dimension again
            noise = torch.reshape(noise, (-1, seqLen, conditioning.shape[2] + data.shape[2], data.shape[3], data.shape[4]))
            predictedNoise = torch.reshape(predictedNoise, (-1, seqLen, conditioning.shape[2] + data.shape[2], data.shape[3], data.shape[4]))

            return noise, predictedNoise, interm_features


        # INFERENCE
        else:
            # conditioned reverse diffusion process
            # print('inference mode of diffusion model')
            dNoise = torch.randn_like(d, device=device)
            cNoise = torch.randn_like(cond, device=device)

            for i in reversed(range(0, self.timesteps)):
                t = i * torch.ones(cond.shape[0], device=device).long()

                # compute conditioned part with normal forward diffusion
                condNoisy = self.sqrtAlphasCumprod[t] * cond + self.sqrtOneMinusAlphasCumprod[t] * cNoise

                dNoiseCond = torch.concat((condNoisy, dNoise), dim=1)

                # backward diffusion process that removes noise to create data

                if (( self.config.model.twin_tower == True) and ( self.config.model.control_connect == True)):
                    unet2_output, intermediate_outputs2 = self.unet2(dNoiseCond, t, context=None)
                    unet1_output, intermediate_outputs1 = self.unet1(dNoiseCond, t, context=intermediate_outputs2['upsample_20'])
                    predictedNoiseCond = unet1_output
                    del unet1_output, unet2_output

                elif (( self.config.model.twin_tower == True) and ( self.config.model.fusion == True)):
                    unet2_output, intermediate_outputs2 = self.unet2(dNoiseCond, t, context=None)
                    unet1_output, intermediate_outputs1 = self.unet1(dNoiseCond, t, context=None)
                    predictedNoiseCond = self.FusionModule(unet1_output, unet2_output, t)
                    del unet1_output, unet2_output

                elif ( self.config.model.twin_tower == False):
                    unet1_output, intermediate_outputs1 = self.unet1(dNoiseCond, t, context=None)
                    predictedNoiseCond = unet1_output
                    del unet1_output
            

                torch.cuda.empty_cache()  # Clear the cache if you're on a GPU



                # use model (noise predictor) to predict mean
                modelMean = self.sqrtRecipAlphas[t] * (dNoiseCond - self.betas[t] * predictedNoiseCond / self.sqrtOneMinusAlphasCumprod[t])

                dNoise = modelMean[:, cond.shape[1]:modelMean.shape[1]] # discard prediction of conditioning
                if i != 0:
                    # sample randomly (only for non-final prediction), use mean directly for final prediction
                    dNoise = dNoise + self.sqrtPosteriorVariance[t] * torch.randn_like(dNoise)


            # once denoising is completing, save interm outputs of the UNet
            interm_features = {'UNet1': intermediate_outputs1,
                               'UNet2': intermediate_outputs2 if self.config.model.twin_tower == True else {}
                               }
            # unstack batch and sequence dimension again
            dNoise = torch.reshape(dNoise, (-1, seqLen, data.shape[2], data.shape[3], data.shape[4]))

            return dNoise, interm_features


class FusionModule(nn.Module):
    def __init__(self, config):
        super(FusionModule, self).__init__()
        self.strategy = config.model.fusion_params.fusion_strategy
        self.time_dim = config.model.time_dim
        self.dim = config.model.dim

        if self.strategy == "cross_attention":
            self.num_heads = config.model.fusion_params.num_heads
            self.embed_dim = config.model.fusion_params.embed_dim
            self.query_proj = nn.Linear(self.embed_dim, self.embed_dim)
            self.key_proj = nn.Linear(self.embed_dim, self.embed_dim)
            self.value_proj = nn.Linear(self.embed_dim, self.embed_dim)
            self.attn = nn.MultiheadAttention(self.embed_dim, self.num_heads)
            
            
        elif self.strategy == "add":
            self.weight_network = nn.Sequential(
                                    nn.Linear(self.time_dim, int(self.time_dim/2)),
                                    nn.ReLU(),
                                    nn.Linear(int(self.time_dim/2), 1)
                                    # nn.Softmax(dim=-1)  # Ensure weights sum to 1
                                    )
        
            self.weight_network.apply(self.initialize_weights)

            # Add other necessary modules as per strategy

            self.time_mlp = nn.Sequential(
                        SinusoidalPositionEmbeddings(self.dim),
                        nn.Linear(self.dim, self.time_dim),
                        nn.GELU(),
                        nn.Linear(self.time_dim, self.time_dim),
                    )
    
    def forward(self, x1, x2, t):
        if self.strategy == "add":
            # Element-wise addition
            t_emb = self.time_mlp(t)
            weights = self.weight_network(t_emb)
            return x1 + weights[:].unsqueeze(1).unsqueeze(2) * x2 * 0.01
        
        elif self.strategy == "cross_attention":
            # Reshape feature maps to (H*W, B, CT) for attention mechanism
            # print('shape of input to fusion layer:', x1.size())

            B, C, H, W = x1.size()
            x1 = x1.view(B, C, -1).permute(2, 0, 1)  # (H*W, B, C)
            x2 = x2.view(B, C, -1).permute(2, 0, 1)  # (H*W, B, C)
            
            # Project queries, keys, and values for cross-attention
            query = self.query_proj(x1)  # (H*W, B, C)
            key = self.key_proj(x2)      # (H*W, B, C)
            value = self.value_proj(x2)  # (H*W, B, C)
            
            # Apply multi-head attention
            attn_output, _ = self.attn(query, key, value)
            
            # Reshape the attention output back to (B, C, H, W)
            fused_output = attn_output.permute(1, 2, 0).view(B, C, H, W)

            return fused_output
        else:
            raise ValueError(f"Unknown fusion strategy: {self.strategy}")

    def initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.constant_(layer.weight, 0)  # Initialize weights to zero
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)  # Initialize biases to zero