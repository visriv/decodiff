from validation import animate
import torch.nn as nn
import torch.nn.functional as F

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
from validation import *
from omegaconf import OmegaConf
import wandb



config = OmegaConf.load('/home/autoreg-pde-diffusion/scripts/kol.yaml')


device = config.device if torch.cuda.is_available() else "cpu"
start_from_checkpoint = config.experiment.start_from_checkpoint
checkpoint_path = config.experiment.checkpoint_path
run_id = config.experiment.run_id
save_dir = './runs/{}/{}'.format(config.experiment.project, run_id)
os.makedirs(save_dir, exist_ok=True)

field = 0


test_set = kolTorchDataset(
                split= "val",                 
                data_path = config.data.data_dirs,
                window_length=config.data.rollout_total_seq_len,
                k=config.data.val_downsample_k,
                train_ratio = config.data.train_ratio,
                val_ratio = config.data.val_ratio,
                standardize=True,
                crop=config.data.crop)
    


testSampler = SequentialSampler(test_set)
testLoader = DataLoader(test_set, sampler=testSampler, batch_size=1, drop_last=False)

with torch.no_grad():
    for s, sample in enumerate(testLoader, 0):
                
        sample = rearrange(sample, 'b t h w c -> b t c h w').unsqueeze(0)
        print('sample shape from testLoader:', sample.cpu().numpy().shape)
            
# [0,0,i,field,:,:]
        animate(sample, 
        os.path.join(save_dir, 'gt_' + config.experiment.save_plot_video_path) + '_.mp4',
        field) 



