
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


class DiffusionDirect(nn.Module):
    def __init__(self, 
                config):
        super(DiffusionDirect, self).__init__()
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

        self.register_buffer('posterior_log_variance_clipped', torch.log(posteriorVariance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphasCumprodPrev) / (1. - alphasCumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphasCumprodPrev) * torch.sqrt(alphas) / (1. - alphasCumprod))
        self.register_buffer('loss_weight', torch.sqrt(alphas) * torch.sqrt(1. - alphasCumprod) / betas / 100)




        # backbone model
        self.unet1 = Unet(
            dim=self.dim,
            channels = self.cond_channels + self.data_channels,
            out_dim = config.model.out_channels,
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
        # print(device)
        seqLen = data.shape[1]

        # combine batch and sequence dimension for decoder processing
        x = torch.reshape(data, (-1, data.shape[2], data.shape[3], data.shape[4]))
        cond = torch.reshape(conditioning, (-1, conditioning.shape[2], conditioning.shape[3], conditioning.shape[4]))

        # TRAINING
        if self.training:

            # forward diffusion process that adds noise to data
            x_and_c = torch.concat((cond, x), dim=1).to(device)
            noise = torch.randn_like(x_and_c, device=device)
            t = torch.randint(0, self.timesteps, (x_and_c.shape[0],), device=device).long()
            x_and_c_noisy = self.sqrtAlphasCumprod[t] * x_and_c + self.sqrtOneMinusAlphasCumprod[t] * noise
 




            if (( self.config.model.twin_tower == True) and ( self.config.model.control_connect == True)):
                unet2_output, intermediate_outputs2 = self.unet2(x_and_c_noisy, t, context=None)
                unet1_output, intermediate_outputs1 = self.unet1(x_and_c_noisy, t, context=intermediate_outputs2['upsample_20'])
                predicted_x = unet1_output
                del unet1_output, unet2_output

            elif (( self.config.model.twin_tower == True) and ( self.config.model.fusion == True)):
                unet2_output, intermediate_outputs2 = self.unet2(x_and_c_noisy, t, context=None)
                unet1_output, intermediate_outputs1 = self.unet1(x_and_c_noisy, t, context=None)
                predicted_x = self.FusionModule(unet1_output, unet2_output, t)
                del unet1_output, unet2_output

            elif ( self.config.model.twin_tower == False):
                unet1_output, intermediate_outputs1 = self.unet1(x_and_c_noisy, t, context=None)
                predicted_x = unet1_output
                del unet1_output

            # print('predicted_x.shape:', predicted_x.shape)

            # Delete tensors if they are no longer needed
            torch.cuda.empty_cache()  # Clear the cache if you're on a GPU

            # once denoising is completing, save interm outputs of the UNet
            interm_features = {'UNet1': intermediate_outputs1,
                               'UNet2': intermediate_outputs2 if self.config.model.twin_tower == True else {}
                    }
            
        
            # unstack batch and sequence dimension again
            noise = torch.reshape(noise, (-1, seqLen, conditioning.shape[2] + data.shape[2], data.shape[3], data.shape[4]))
            x = torch.reshape(x, (-1, seqLen, data.shape[2], data.shape[3], data.shape[4]))
            predicted_x = torch.reshape(predicted_x, (-1, seqLen, data.shape[2], data.shape[3], data.shape[4]))
            # print('predicted_x.shape:', predicted_x.shape)

            return noise, predicted_x, x, self.loss_weight[t], interm_features


        # INFERENCE
        else:
            # conditioned reverse diffusion process
            # print('inference mode of diffusion model')
            xT = torch.randn_like(x, device=device)
            cNoise = torch.randn_like(cond, device=device)
            xt = xT
            # print('cond.shape:', cond.shape)
            # print('dummy text')
            # print('xt.shape:', xt.shape)

            for i in reversed(range(0, self.timesteps)):

                t = i * torch.ones(cond.shape[0], device=device).long()

                # compute conditioned part with normal forward diffusion
                condNoisy = self.sqrtAlphasCumprod[t] * cond + self.sqrtOneMinusAlphasCumprod[t] * cNoise

                xt_condNoisy = torch.concat((condNoisy, xt), dim=1)
                # print('xt_condNoisy.shape:', xt_condNoisy.shape)

                # backward diffusion process that removes noise to create data

                if (( self.config.model.twin_tower == True) and ( self.config.model.control_connect == True)):
                    unet2_output, intermediate_outputs2 = self.unet2(xt_condNoisy, t, context=None)
                    unet1_output, intermediate_outputs1 = self.unet1(xt_condNoisy, t, context=intermediate_outputs2['upsample_20'])
                    x0_hat = unet1_output
                    del unet1_output, unet2_output

                elif (( self.config.model.twin_tower == True) and ( self.config.model.fusion == True)):
                    unet2_output, intermediate_outputs2 = self.unet2(xt_condNoisy, t, context=None)
                    unet1_output, intermediate_outputs1 = self.unet1(xt_condNoisy, t, context=None)
                    x0_hat = self.FusionModule(unet1_output, unet2_output, t)
                    del unet1_output, unet2_output

                elif ( self.config.model.twin_tower == False):
                    unet1_output, intermediate_outputs1 = self.unet1(xt_condNoisy, t, context=None)
                    x0_hat = unet1_output
                    del unet1_output
            

                torch.cuda.empty_cache()  # Clear the cache if you're on a GPU



                # use model (noise predictor) to predict mean
                # modelMean = self.sqrtRecipAlphas[t] * (xt_condNoisy - self.betas[t] * predictedNoiseCond / self.sqrtOneMinusAlphasCumprod[t])
                modelMean_t_first_term = self.posterior_mean_coef1[t] * x0_hat
                modelMean_t_second_term = self.posterior_mean_coef1[t] * xt
                modelMean_t = modelMean_t_first_term + modelMean_t_second_term


                x_tminus1 = modelMean_t#[:, cond.shape[1]:modelMean_t.shape[1]] # discard prediction of conditioning
                
                if i != 0:
                    # sample randomly (only for non-final prediction), use mean directly for final prediction
                    x_tminus1 = x_tminus1 + self.sqrtPosteriorVariance[t] * torch.randn_like(x_tminus1)
                
                xt = x_tminus1
            # end of loop, get the final estimated denoised sample
            x0 = xt
            # once denoising is completing, save interm outputs of the UNet
            interm_features = {'UNet1': intermediate_outputs1,
                               'UNet2': intermediate_outputs2 if self.config.model.twin_tower == True else {}
                               }
            # unstack batch and sequence dimension again
            x0 = torch.reshape(x0, (-1, seqLen, data.shape[2], data.shape[3], data.shape[4]))

            return x0, interm_features


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