import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from einops import rearrange
from functools import partial

import matplotlib.pyplot as plt
import numpy as np



from network import *
from diffusion import *
from dataloader import *
from kol_dataloader import kolTorchDataset

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
    gtPred = np.concatenate([gt[:,sequence,timeSteps,field], predPart[:,sequence,timeSteps,field]])
    print('gt concat shape:', gt[:,sequence,timeSteps,field].shape)
    print('predPart concat shape:', predPart[:,sequence,timeSteps,field].shape)
    print('gtPred shape:', gtPred.shape)

    fig, axs = plt.subplots(nrows=gtPred.shape[0], ncols=gtPred.shape[1], figsize=(gtPred.shape[1]*1.9, gtPred.shape[0]), dpi=150, squeeze=False)

    for i in range(gtPred.shape[0]):
        for j in range(gtPred.shape[1]):
            if i == gtPred.shape[0]-1:
                axs[i,j].set_xlabel("$t=%s$" % (timeSteps[j]+1), fontsize=10)
            if j == 0:
                if i == 0:
                    axs[i,j].set_ylabel("Ground\nTruth", fontsize=10)
                else:
                    axs[i,j].set_ylabel("ACDM\nSample %d" % i, fontsize=10)
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])
            im = axs[i,j].imshow(gtPred[i][j], interpolation="catrom", cmap="RdBu_r")

    fig.savefig(os.path.join(save_dir, config.experiment.plot_file_name) + '_{}.png'.format(epoch))
    plt.show()

    if (config.experiment.plot_video):
        animate(gt, os.path.join(save_dir, 'gt_' + config.experiment.save_plot_video_path) + '_.mp4',
                field) 
        animate(pred, os.path.join(save_dir, 'pred_' +config.experiment.save_plot_video_path) + '_.mp4',
                field) 

 
def get_validation_dataloader(batch_size, rank, world_size, config):
    test_set = kolTorchDataset(
                split= "val",                 
                data_path = config.data.data_dirs,
                window_length=config.data.rollout_total_seq_len,
                k=config.data.val_downsample_k,
                train_ratio = config.data.train_ratio,
                val_ratio = config.data.val_ratio,
                standardize=True,
                crop=config.data.crop)
    sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(test_set, batch_size=1, sampler=sampler)
    return dataloader





def validation_step(model, config, rank, save_dir, epoch):
    world_size = torch.cuda.device_count()

    if rank != 0:
        return  # Only the main process performs validation
    print(f"Process {rank}: Starting validation at epoch {epoch}")

    if config.experiment.train == False:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)


    device = "cuda" if torch.cuda.is_available() else "cpu"

    numSamples = 1

    
    try: # load model if not trained/finetuned above
        model
    except NameError:
        
        model = DiffusionModel(config)

        # load weights from checkpoint
        loaded = torch.load(config.experiment.checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(loaded["stateDictDecoder"])
    
    model.to(rank)
    model.eval()
    val_loader = get_validation_dataloader(config.training.batch_size, rank, world_size, config)
   



    # sampling loop
    print("\nStarting sampling...")
    gt = []
    pred = []
    with torch.no_grad():
        for s, sample in enumerate(val_loader, 0):
            if (s == config.validation.n_val_samples):
                break
            sample = rearrange(sample, 'b t h w c -> b t c h w').to(rank)
            print('sample shape from testLoader:', sample.unsqueeze(0).cpu().numpy().shape)
            gt += [sample.unsqueeze(0).cpu().numpy()]
            d = sample.to(rank).repeat(numSamples,1,1,1,1) # reuse batch dim for samples
            print('d shape from testLoader:', d.shape)

            prediction = torch.zeros_like(d, device=rank)
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
                
                print(prediction.size(), len(result), result.shape,  cond.shape)
                print(type(prediction), type(result))
                # print(result[0], result[1])
                # result[:,:,-len(simParams):] = d[:,i:i+1,-len(simParams):] # replace simparam prediction with true values
                prediction[:,i:i+1] = result.to(rank)

            prediction = torch.reshape(prediction, (numSamples, -1, d.shape[1], d.shape[2], d.shape[3], d.shape[4]))
            pred += [prediction.cpu().numpy()]
            print("  Sequence %d finished" % s)
            
            del prediction, sample, d, result, cond
            torch.cuda.empty_cache()



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
    
    del roll_val_mse, roll_val_mae
    
    visualize(pred, gt, config, save_dir, epoch)

    model.train()
    return pred, gt