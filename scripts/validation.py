import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler

from einops import rearrange
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import os


from src.model.network import *
from src.model.diffusion import *
# from dataloader import *
from src.kol_dataloader import kolTorchDataset

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
    print(mae.shape)
    gtPredMae = np.concatenate([gt[:,sequence,timeSteps,field], 
                             predPart[:,sequence,timeSteps,field],
                             mae[:,sequence,timeSteps,field]])
    
    # for t in range(gt.shape[2]):
    #     # Extract the slices for the current timestamp
    #     tensor1_t = torch.from_numpy(gt[0,0,t,0,:,:])
    #     tensor2_t = torch.from_numpy(pred[0,0,t,0,:,:])
        
    #     # Compute MSE and MAE for the current timestamp
    #     mse_t = F.mse_loss(tensor1_t, tensor2_t)
    #     mae_t = F.l1_loss(tensor1_t, tensor2_t)


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
            im = axs[i,j].imshow(gtPredMae[i][j], interpolation="catrom", cmap="RdBu_r")

    cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('velocity component', fontsize=10)

    fig.savefig(os.path.join(save_dir, config.experiment.plot_file_name) + '_{}.png'.format(epoch))
    plt.show()

    if (config.experiment.plot_video):
        animate(gt, os.path.join(save_dir, 'gt_' + config.experiment.save_plot_video_path) + '_.mp4',
                field) 
        animate(pred, os.path.join(save_dir, 'pred_' +config.experiment.save_plot_video_path) + '_.mp4',
                field) 

    # now plot the spatial frequency spectrum
    print('plotting the spatial spectrum')


    # Assuming gt and pred are numpy arrays of shape [1, 1, T, 1, H, W]
    # Extract the relevant dimensions (T, H, W)
    np.save(os.path.join(save_dir, 'gtNpy.npy'), gt )
    np.save(os.path.join(save_dir, 'predNpy.npy'), pred )

    T = gt.shape[2]
    H, W = gt.shape[4], gt.shape[5]

    fig_new, axs = plt.subplots(nrows=2, ncols=T, figsize=(T * 5, 2 * 4), dpi=150, squeeze=False)

    # Loop through each timestamp to calculate and plot the combined energy spectrum
    for t in range(T):
        # Extract gt and pred arrays at timestamp t
        gt_t = gt[0, 0, t, 0, :, :]
        pred_t = pred[0, 0, t, 0, :, :]
        
        # Round up the size along each axis to an even number
        n_H = int(math.ceil(H / 2.) * 2)
        n_W = int(math.ceil(W / 2.) * 2)
        
        # Compute the 2D Fourier transform for gt and pred using rfft along both axes
        fft_gt_x = np.fft.rfft(gt_t, n=n_W, axis=1)
        fft_gt_y = np.fft.rfft(gt_t, n=n_H, axis=0)

        fft_pred_x = np.fft.rfft(pred_t, n=n_W, axis=1)
        fft_pred_y = np.fft.rfft(pred_t, n=n_H, axis=0)
        
        # Compute power spectrum (multiply by complex conjugate)
        energy_gt_x = fft_gt_x.real ** 2 + fft_gt_x.imag ** 2
        energy_gt_y = fft_gt_y.real ** 2 + fft_gt_y.imag ** 2

        energy_pred_x = fft_pred_x.real ** 2 + fft_pred_x.imag ** 2
        energy_pred_y = fft_pred_y.real ** 2 + fft_pred_y.imag ** 2

        # Average over appropriate axes
        energy_gt_x = energy_gt_x.sum(axis=0) / fft_gt_x.shape[0]
        energy_gt_y = energy_gt_y.sum(axis=1) / fft_gt_y.shape[1]

        energy_pred_x = energy_pred_x.sum(axis=0) / fft_pred_x.shape[0]
        energy_pred_y = energy_pred_y.sum(axis=1) / fft_pred_y.shape[1]

        # Combine energies for a single spectrum
        energy_gt = 0.5 * (energy_gt_x + energy_gt_y)
        energy_pred = 0.5 * (energy_pred_x + energy_pred_y)

        # Generate wavenumber axis (only for positive frequencies)
        wavenumber_x = np.fft.rfftfreq(n_W)
        wavenumber_y = np.fft.rfftfreq(n_H)

        # Since energy_gt and energy_pred are averaged over x and y axes, 
        # we use the wavenumber from one of the axes (they represent equivalent ranges)
        wavenumber = wavenumber_x  # or wavenumber_y, since the final energy spectrum is averaged

        # Sort wavenumber and energy values in descending order
        sorted_indices = np.argsort(wavenumber)[::-1]
        wavenumber_sorted = wavenumber[sorted_indices]
        energy_gt_sorted = energy_gt[sorted_indices]
        energy_pred_sorted = energy_pred[sorted_indices]

        # Plotting the combined energy spectrum for gt and pred (linear scale)
        axs[0, t].plot(wavenumber_sorted, energy_gt_sorted, label='GT', color='blue')
        axs[0, t].plot(wavenumber_sorted, energy_pred_sorted, label='Pred', color='red')
        axs[0, t].set_title(f"Energy Spectrum (Linear Scale) (t={t+1})")
        axs[0, t].set_xlabel('Wavenumber')
        axs[0, t].set_ylabel('Amplitude')
        axs[0, t].grid(True)
        axs[0, t].legend()

        # Plotting the combined energy spectrum for gt and pred (log-log scale)
        axs[1, t].plot(wavenumber_sorted, energy_gt_sorted, label='GT', color='blue')
        axs[1, t].plot(wavenumber_sorted, energy_pred_sorted, label='Pred', color='red')
        axs[1, t].set_xscale('log')
        axs[1, t].set_yscale('log')
        axs[1, t].set_title(f"Energy Spectrum (Log-Log Scale) (t={t+1})")
        axs[1, t].set_xlabel('Wavenumber')
        axs[1, t].set_ylabel('Amplitude')
        axs[1, t].grid(True)
        axs[1, t].legend()

    # Adjust layout for better viewing
    plt.tight_layout()
    fig_new.savefig(os.path.join(save_dir, config.experiment.freq_plot_file_name) + '_{}.png'.format(epoch))

    plt.show()




    


 




def validation_step(model, config, save_dir, epoch):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    numSamples = 1

    diffusionSteps = config.model.diffusion_steps

    try: # load model if not trained/finetuned above
        model
    except NameError:
        data_channels = config.model.data_channels
        cond_channels = config.model.input_steps * data_channels #2 * (2 + len(sim_fields) + len(sim_params))

        model = DiffusionModel(diffusionSteps, cond_channels, data_channels)

        # load weights from checkpoint
        loaded = torch.load(config.experiment.checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(loaded["stateDictDecoder"])
    model.eval()
    model.to(device)

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

    # sampling loop
    print("\nStarting sampling...")
    gt = []
    pred = []
    with torch.no_grad():
        for s, sample in enumerate(testLoader, 0):
            if (s == config.validation.n_val_samples):
                break
            sample = rearrange(sample, 'b t h w c -> b t c h w')
            print('sample shape from testLoader:', sample.unsqueeze(0).cpu().numpy().shape)
            gt += [sample.unsqueeze(0).cpu().numpy()]
            d = sample.to(device).repeat(numSamples,1,1,1,1) # reuse batch dim for samples
            print('d shape from testLoader:', d.shape)

            prediction = torch.zeros_like(d, device=device)
            inputSteps = config.model.input_steps

            for i in range(inputSteps): # no prediction of first steps
                prediction[:,i] = d[:,i]

            for i in range(inputSteps, d.shape[1]):
                print('{} steps out of'.format(i), d.shape[1])
                cond = []
                for j in range(inputSteps,0,-1):
                    cond += [prediction[:, i-j : i-(j-1)]] # collect input steps
                cond = torch.concat(cond, dim=2) # combine along channel dimension

                result = model(conditioning=cond, data=d[:,i-1:i]) # auto-regressive inference
                
                


                # result[:,:,-len(simParams):] = d[:,i:i+1,-len(simParams):] # replace simparam prediction with true values
                # print('shape of result from model(): [len(result), result[0].shape]', len(result), result[0].shape)
                prediction[:,i:i+1] = result

            prediction = torch.reshape(prediction, (numSamples, -1, d.shape[1], d.shape[2], d.shape[3], d.shape[4]))
            pred += [prediction.cpu().numpy()]
            print("  Sequence %d finished" % s)

    # Clear CUDA memory
    torch.cuda.empty_cache()
    gc.collect()
    print("Sampling complete!\n")

    gt = np.concatenate(gt, axis=1)
    pred = np.concatenate(pred, axis=1)

    # undo data normalization
    # normMean = test_set.transform.normMean[[0,1,2,3,5]]
    # normStd = test_set.transform.normStd[[0,1,2,3,5]]
    # normMean = np.expand_dims(normMean, axis=(0,1,2,4,5))
    # normStd = np.expand_dims(normStd, axis=(0,1,2,4,5))
    # gt = (gt * normStd) + normMean
    # pred = (pred * normStd) + normMean

    print("Ground truth and prediction tensor with shape:")
    print("(samples, sequences, sequenceLength, channels, sizeX, sizeY)")
    print("GT: %s" % str(gt.shape))
    print("Prediction: %s" % str(pred.shape))

    roll_val_t_mse = []
    roll_val_t_mae = []

    for t in range(gt.shape[2]):
        # Extract the slices for the current timestamp
        tensor1_t = torch.from_numpy(gt[0,0,t,0,:,:])
        tensor2_t = torch.from_numpy(pred[0,0,t,0,:,:])
        
        # Compute MSE and MAE for the current timestamp
        mse_t = F.mse_loss(tensor1_t, tensor2_t)
        mae_t = F.l1_loss(tensor1_t, tensor2_t)
        
        # Append the loss values to the respective lists
        roll_val_t_mse.append(mse_t.item())
        roll_val_t_mae.append(mae_t.item())

        wandb.log({'rollout_val_t_mse': mse_t.item(), 
                   'rollout_val_t_mae': mae_t.item(),
                   'timestep': t})

    roll_val_mse = F.mse_loss(torch.from_numpy(gt[0,0,:,0,:,:]),
                              torch.from_numpy(pred[0,0,:,0,:,:]))
    roll_val_mae = F.l1_loss(torch.from_numpy(gt[0,0,:,0,:,:]),
                              torch.from_numpy(pred[0,0,:,0,:,:]))

    wandb.log({'roll_val_mse': roll_val_mse, 
               'roll_val_mae': roll_val_mae
                })
    
    visualize(pred, gt, config, save_dir, epoch)
    return pred, gt