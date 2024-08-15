import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler

from einops import rearrange
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

import math
import os, json
from typing import List, Tuple, Dict

from network import *
from diffusion import *
from dataloader import *
from kol_dataloader import kolTorchDataset
from validation import validation_step
from omegaconf import OmegaConf
import wandb
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set to "0" to use only the first GPU, "1" for the second GPU, etc.

def load_checkpoint(model, optimizer, epoch):
    file_path = 'model_checkpoint_{:02d}.pth'.format(epoch)  # Formatted filename with epoch number
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['stateDictDecoder'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

config = OmegaConf.load('/home/autoreg-pde-diffusion/scripts/kol.yaml')

# model definition
data_channels = config.model.data_channels
cond_channels = config.model.input_steps * data_channels #2 * (2 + len(sim_fields) + len(sim_params))
diffusion_steps = config.model.diffusion_steps
checkpoint_path = config.experiment.checkpoint_path

model = DiffusionModel(diffusion_steps, cond_channels, data_channels)
loaded = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(loaded["stateDictDecoder"])

run_id = config.experiment.run_id
save_dir = './runs/{}/{}'.format(config.experiment.project, run_id)
os.makedirs(save_dir, exist_ok=True)

epoch = 'last'

validation_step(model, config, save_dir, epoch)
