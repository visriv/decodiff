import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from einops import rearrange
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import os



from src.model.network import *
from src.model.diffusion import *
from src.kol_dataloader import kolTorchDataset
from src.utils.get_model import get_model
from src.utils.visualize_on_unet import *
from src.utils.spectrum import get_spatial_spectrum

import gc
import matplotlib.animation as animation
import wandb

def animate(video_array, save_plot_video_path,
            field):

    frames = [] # for storing the generated images
    fig = plt.figure()
    for i in range(video_array.shape[2]):
        # frames.append([plt.imshow(video_array[0][i][:,:,0], cmap='RdBu_r',animated=True)])
        frames.append([plt.imshow(video_array[0,0,i,field,:,:], cmap='RdBu_r',animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                    repeat_delay=1000)
    ani.save(save_plot_video_path)
    plt.show()

def visualize(pred, gt, config, save_dir, epoch):
    sequence = 0
    samples = [0]
    timeSteps = [i * gt.shape[2] // 10 for i in range(10)] + [gt.shape[2]-1]

    field = 0 # velocity_x (0), velocity_y (1), density (2), or pressure (3)

    predPart = pred[samples]

    mae = np.abs(gt - predPart)
    # print(mae.shape)

    gtPredMae = np.concatenate([gt[:,sequence,timeSteps,field], 
                             predPart[:,sequence,timeSteps,field],
                             mae[:,sequence,timeSteps,field]])


    vmin = -2.5
    vmax = 2.5


    print('gt concat shape:', gt[:,sequence,timeSteps,field].shape)
    print('predPart concat shape:', predPart[:,sequence,timeSteps,field].shape)
    print('gtPredMae shape:', gtPredMae.shape)

    fig, axs = plt.subplots(nrows=gtPredMae.shape[0], ncols=gtPredMae.shape[1], figsize=(gtPredMae.shape[1]*1.9, gtPredMae.shape[0]), dpi=150, squeeze=False)

    for i in range(gtPredMae.shape[0]):
        for j in range(gtPredMae.shape[1]):
            if i == gtPredMae.shape[0]-1:
                axs[i,j].set_xlabel("$t=%s$" % (timeSteps[j]+1), fontsize=10)
            if j == 0:
                if i == 0:
                    axs[i,j].set_ylabel("Ground\nTruth", fontsize=10)
                elif i == gtPredMae.shape[0]-1:
                    axs[i,j].set_ylabel("Mean Absolute\nError", fontsize=10)
                else:
                    axs[i,j].set_ylabel("ACDM\nSample %d" % i, fontsize=10)
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])
            im = axs[i,j].imshow(gtPredMae[i][j], interpolation="catrom", cmap="RdBu_r",
                                 vmin=vmin, vmax=vmax)

    cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('velocity component', fontsize=10)

    fig.savefig(os.path.join(save_dir, config.experiment.plot_file_name) + '_{}.png'.format(epoch))
    plt.show()

    if (config.experiment.plot_video):
        animate(gt, os.path.join(save_dir, 'gt_' + config.experiment.save_plot_video_path) + '_{}.mp4'.format(epoch),
                field) 
        animate(pred, os.path.join(save_dir, 'pred_' +config.experiment.save_plot_video_path) + '_{}.mp4'.format(epoch),
                field) 


def plot_spectrum(gt, pred, save_dir, config, epoch):
    # now plot the spatial frequency spectrum
    print('plotting the spatial spectrum')


    # Assuming gt and pred are numpy arrays of shape [1, 1, T, 1, H, W]
    # Extract the relevant dimensions (T, H, W)
    np.save(os.path.join(save_dir, '_{}_gtNpy.npy'.format(epoch)), gt )
    np.save(os.path.join(save_dir, '_{}_predNpy.npy'.format(epoch)), pred )

    timeSteps = [i * gt.shape[2] // config.experiment.n_timestamps_plot for i in range(config.experiment.n_timestamps_plot)] + [gt.shape[2]-1]
    T = len(timeSteps)


    fig_new, axs = plt.subplots(nrows=2, ncols=T, figsize=(T * 5, 2 * 4), dpi=150, squeeze=False)

    # Loop through each timestamp to calculate and plot the combined energy spectrum
    t = 0
    for tau in timeSteps:
        # Extract gt and pred arrays at timestamp t
        gt_t = gt[0, 0, tau, 0, :, :]
        pred_t = pred[0, 0, tau, 0, :, :]




        energy_gt_sorted, wavenumber_sorted = get_spatial_spectrum(gt_t)
        energy_pred_sorted, wavenumber_sorted = get_spatial_spectrum(pred_t)

        np.save(os.path.join(save_dir, 'wavenumber_sorted.npy'), wavenumber_sorted )
        np.save(os.path.join(save_dir, 'energy_gt_sorted.npy'), energy_gt_sorted )
        np.save(os.path.join(save_dir, 'energy_pred_sorted.npy'), energy_pred_sorted )

        # Plotting the combined energy spectrum for gt and pred (linear scale)

        axs[0, t].plot(wavenumber_sorted, energy_gt_sorted, label='GT', color='blue')
        axs[0, t].plot(wavenumber_sorted, energy_pred_sorted, label='Pred', color='red')
        axs[0, t].set_title(f"Energy Spectrum (Linear Scale) (t={tau+1})")
        axs[0, t].set_xlabel('Wavenumber')
        axs[0, t].set_ylabel('Amplitude')
        axs[0, t].grid(True)
        axs[0, t].legend()

        # Plotting the combined energy spectrum for gt and pred (log-log scale)
        axs[1, t].plot(wavenumber_sorted, energy_gt_sorted, label='GT', color='blue')
        axs[1, t].plot(wavenumber_sorted, energy_pred_sorted, label='Pred', color='red')
        axs[1, t].set_xscale('log')
        axs[1, t].set_yscale('log')
        axs[1, t].set_title(f"Energy Spectrum (Log-Log Scale) (t={tau+1})")
        axs[1, t].set_xlabel('Wavenumber')
        axs[1, t].set_ylabel('Amplitude')
        axs[1, t].grid(True)
        axs[1, t].legend()
        t += 1

    # Adjust layout for better viewing
    plt.tight_layout()
    fig_new.savefig(os.path.join(save_dir, config.experiment.freq_plot_file_name) + '_{}.png'.format(epoch))

    plt.show()