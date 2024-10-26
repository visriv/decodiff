import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from einops import rearrange

import numpy as np
from tqdm import tqdm






from src.model.network import *
from src.model.diffusion import *
from src.kol_dataloader import kolTorchDataset
from src.utils.get_model import get_model
from src.utils.visualize_on_unet import *
from src.utils.visualize_2d_plots import *
from src.utils.log_wandb import log_error
import gc
import wandb




 
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
        
        model = get_model(config).to(rank)
        # load weights from checkpoint
        loaded = torch.load(config.experiment.checkpoint_path, map_location=torch.device('cpu'))
        new_state_dict = {k.replace('module.', ''): v for k, v in loaded['stateDictDecoder'].items()}
        loaded['stateDictDecoder'] = new_state_dict
        # print(loaded['stateDictDecoder'].keys())
        model.load_state_dict(loaded["stateDictDecoder"],  strict=False)

    
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

            for i in tqdm(range(inputSteps, d.shape[1]), desc="autoreg rollout steps"):
                print('{} steps out of'.format(i), d.shape[1])
                cond = []
                for j in range(inputSteps,0,-1):
                    cond += [prediction[:, i-j : i-(j-1)]] # collect input steps
                cond = torch.concat(cond, dim=2) # combine along channel dimension

                result, interm_features = model(conditioning=cond, data=d[:,i-1:i]) # auto-regressive inference
                
                prediction[:,i:i+1] = result.to(rank)

            prediction = torch.reshape(prediction, (numSamples, -1, d.shape[1], d.shape[2], d.shape[3], d.shape[4]))
            pred += [prediction.cpu().numpy()]
            print("  Sequence %d finished" % s)
            
            del prediction, sample, d, result, cond
            torch.cuda.empty_cache()

    # interm_features['gt'] = gt[0, 0, -1, 0, :, :] #save the t = last GT for visualization
    # visualize intermediate embeddings
    np.save(save_dir + '/interm_embeds.npy', interm_features)
    visualize_interm_embeds(interm_features, save_dir, config)
    visualize_spatial_spectra(interm_features, save_dir, config)

    # Clear CUDA memory
    torch.cuda.empty_cache()
    gc.collect()
    print("Sampling complete!\n")

    gt = np.concatenate(gt, axis=1)
    pred = np.concatenate(pred, axis=1)
    print("Ground truth and prediction tensor with shape:")
    print("(samples, sequences, sequenceLength, channels, sizeX, sizeY)")
    print("GT: %s" % str(gt.shape))
    print("Prediction: %s" % str(pred.shape))

    log_error(gt, pred)
    
    visualize(pred, gt, config, save_dir, epoch)
    plot_spectrum(gt, pred, save_dir, config, epoch)


    model.train()
    return pred, gt