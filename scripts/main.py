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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  

def save_checkpoint(model, optimizer, epoch, file_path):
    checkpoint = {
        'epoch': epoch,
        'stateDictDecoder': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, file_path)


def load_checkpoint(model, optimizer, epoch):
    file_path = 'model_checkpoint_{:02d}.pth'.format(epoch)  # Formatted filename with epoch number
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['stateDictDecoder'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch




# wandb.login(force=True)
wandb.init(project="diffore", entity="visriv")

# Load configuration from YAML file
config = OmegaConf.load('/home/autoreg-pde-diffusion/scripts/kol.yaml')

device = config.device if torch.cuda.is_available() else "cpu"
print("Training device: %s" % device)

start_from_checkpoint = config.experiment.start_from_checkpoint
checkpoint_path = config.experiment.checkpoint_path
run_id = config.experiment.run_id
save_dir = './runs/{}/{}'.format(config.experiment.project, run_id)
os.makedirs(save_dir, exist_ok=True)

save_ckpt_every_n_epochs = config.experiment.save_ckpt_every_n_epochs
val_every_n_epochs = config.experiment.val_every_n_epochs


batch_size = config.training.batch_size
if start_from_checkpoint:
    epochs = config.training.finetune_epochs
    lr = config.training.finetune_learning_rate
else:
    epochs = config.training.epochs
    lr = config.training.learning_rate

# sequence_length = config.sequence_length
# sim_fields = config.sim_fields
# sim_params = config.sim_params
diffusion_steps = config.model.diffusion_steps

train_set = kolTorchDataset(
                split= "train",                 
                data_path = config.data.data_dirs,
                window_length=config.data.total_seq_len,
                train_ratio = config.data.train_ratio,
                val_ratio = config.data.val_ratio,
                standardize=True,
                crop=config.data.crop)



train_loader = DataLoader(
    train_set, 
    batch_size=batch_size, 
    drop_last=True, 
    num_workers=config.dataloader.num_workers
)

# model definition
data_channels = config.model.data_channels
cond_channels = config.model.input_steps * data_channels #2 * (2 + len(sim_fields) + len(sim_params))


model = DiffusionModel(diffusion_steps, cond_channels, data_channels)

if start_from_checkpoint:
    # load weights from checkpoint
    loaded = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(loaded["stateDictDecoder"])
model.train()
model.to(device)

# print model info and trainable weights
params_trainable = sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])
params = sum([np.prod(p.size()) for p in model.parameters()])
#print(model)
print("Trainable Weights (All Weights): %d (%d)" % (params_trainable, params))

epoch = 'last'
if config.experiment.train:
    # training loop
    print("\nStarting training...")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        losses = []
        for s, sample in enumerate(train_loader, 0):
            optimizer.zero_grad()
            print(s)
            sample = rearrange(sample, 'b t h w c -> b t c h w')
            d = sample.to(device)

            input_steps = config.model.input_steps
            cond = []
            for i in range(input_steps):
                cond += [d[:,i:i+1]] # collect input steps
            conditioning = torch.concat(cond, dim=2) # combine along channel dimension
            data = d[:, input_steps:input_steps+1]

            noise, predicted_noise = model(conditioning=conditioning, data=data)

            loss = F.smooth_l1_loss(noise, predicted_noise)
            print("    [Epoch %2d, Batch %4d]: %1.7f" % (epoch, s, loss.detach().cpu().item()))
            loss.backward()

            losses += [loss.detach().cpu().item()]

            optimizer.step()
        print("[Epoch %2d, FULL]: %1.7f" % (epoch, sum(losses)/len(losses)))
        wandb.log({"train_loss": sum(losses)/len(losses), "epoch": epoch})
        # Specify the path to save the checkpoint
        if (epoch % save_ckpt_every_n_epochs == 0):
            file_path = os.path.join(save_dir, 'model_checkpoint_{:02d}.pth'.format( epoch))
            save_checkpoint(model, optimizer, epoch=epoch, file_path=file_path)

        if ((epoch+1) % val_every_n_epochs == 0):
            validation_step(model, config, save_dir, epoch)

        # Clear CUDA memory
        torch.cuda.empty_cache()
        gc.collect()

    print("Training complete!")


if config.experiment.validate:
    pred, gt = validation_step(model, config, save_dir, epoch)


if config.experiment.analysis:
    gtTemp = gt[:,:,:,0:1] # take only one chanel 
    predTemp = pred[:,:,:,0:1]

    print('gtTemp shape:', gtTemp.shape)
    diffGt = np.abs( gtTemp[:,:,1:gtTemp.shape[2]-1] - gtTemp[:,:,2:gtTemp.shape[2]])
    diffGt = np.mean(diffGt, axis=(3,4,5)) # channel-wise and spatial mean
    minGt = np.min(diffGt, axis=(0,1)) # lower bound over sequences
    maxGt = np.max(diffGt, axis=(0,1)) # upper bound over sequences
    meanGt = np.mean(diffGt, axis=(0,1)) # sample- and sequence mean

    diffPred = np.abs( predTemp[:,:,1:predTemp.shape[2]-1] - predTemp[:,:,2:predTemp.shape[2]])
    diffPred = np.mean(diffPred, axis=(3,4,5)) # channel-wise and spatial mean
    minPred = np.min(diffPred, axis=(0,1)) # lower bound over samples and sequences
    maxPred = np.max(diffPred, axis=(0,1)) # upper bound over samples and sequences
    meanPred = np.mean(diffPred, axis=(0,1)) # sample- and sequence mean


    fig, ax = plt.subplots(1, figsize=(5,2), dpi=150)
    ax.set_title("Temporal Stability")
    ax.set_ylabel("$\Vert \, (s^{t} - s^{t-1}) / \Delta t \, \Vert_1$")
    ax.yaxis.grid(True)
    ax.set_xlabel("Time step $t$")

    ax.plot(np.arange(meanGt.shape[0]), meanGt, color="k", label="Simulation", linestyle="dashed")
    ax.fill_between(np.arange(meanGt.shape[0]), minGt, maxGt, facecolor="k", alpha=0.15)

    ax.plot(np.arange(meanPred.shape[0]), meanPred, color="tab:orange", label="ACDM")
    ax.fill_between(np.arange(meanPred.shape[0]), minPred, maxPred, facecolor="tab:orange", alpha=0.15)

    fig.legend()
    plt.show()
    plt.savefig(os.path.join(save_dir, config.experiment.stability_file_name) + '_{}.png'.format(epoch))





    sequence = 0
    fracX = 0.25 # closely behing the cylinder
    fracY = 0.5 # vertically centered
    field = 0 # velocity_x (0), velocity_y (1), density (2), or pressure (3)

    posX = int(fracX * gt.shape[4])
    posY = int(fracY * gt.shape[5])

    gtPred = np.concatenate([gt[:,sequence,:,field, posX, posY], pred[:,sequence,:,field, posX, posY]])

    fft = np.fft.fft(gtPred, axis=1)
    fft = np.real(fft * np.conj(fft))
    n = fft.shape[1]
    gridSpacing = 0.002 # delta t between frames from simulation
    freq = np.fft.fftfreq(n, d=gridSpacing)[1:int(n/2)]
    fft = fft[:,1:int(n/2)] # only use positive fourier frequencies

    gtFFT = fft[0]
    minPredFFT = np.min(fft[1:], axis=0) # lower bound over samples
    maxPredFFT = np.max(fft[1:], axis=0) # upper bound over samples
    meanPredFFT = np.mean(fft[1:], axis=0) # sample mean


    # plot eval point
    fig, ax = plt.subplots(1, figsize=(5,2), dpi=150)
    ax.set_title("Evaluation Point")
    ax.imshow(gt[0,sequence,0,field], interpolation="catrom", cmap="RdBu_r")
    ax.scatter(posX, posY, s=200, color="red", marker="x", linewidth=2)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    plt.savefig(os.path.join(save_dir, config.experiment.eval_node) + '_{}.png'.format(epoch))


    # plot spectral analysis
    fig, ax = plt.subplots(1, figsize=(5,2), dpi=150)
    ax.set_title("Spectral Analysis")
    ax.set_xlabel("Temporal frequency $f$ (at point downstream)")
    ax.set_ylabel("Amplitude $*f^2$")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.yaxis.grid(True)

    ax.plot(freq, gtFFT * (freq**2), color="k", label="Simulation", linestyle="dashed")

    ax.plot(freq, meanPredFFT * (freq**2), color="tab:orange", label="ACDM")
    ax.fill_between(freq, minPredFFT * (freq**2), maxPredFFT * (freq**2), facecolor="tab:orange", alpha=0.15)

    fig.legend()
    plt.show()
    plt.savefig(os.path.join(save_dir, config.experiment.spectral_analysis) + '_{}.png'.format(epoch))
