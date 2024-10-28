import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import argparse


import os
# os.environ['PYTHONPATH'] = '/home/users/nus/e1333861/decodiff:./'
from src.model.network import *
from src.model.diffusion import *
from src.model.pde_refiner import *
from src.kol_dataloader import kolTorchDataset
from validation_distri import validation_step
from src.utils.optim import warmup_lambda, torchOptimizerClass
from src.utils.get_model import get_model
from omegaconf import OmegaConf
import wandb
from einops import reduce
from tqdm import tqdm
import re

os.environ['MASTER_ADDR'] = 'localhost' 
os.environ['MASTER_PORT'] = '12365'   
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

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
    model.load_state_dict(checkpoint['stateDictDecoder'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch



def setup(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()


def get_dataloader(batch_size, rank, world_size, config):
    train_set = kolTorchDataset(
                split= "train",                 
                data_path = config.data.data_dirs,
                window_length=config.data.total_seq_len,
                train_ratio = config.data.train_ratio,
                val_ratio = config.data.val_ratio,
                standardize=True,
                crop=config.data.crop)
    sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
    return dataloader, train_set





def main_child(rank, world_size, config):
    setup(rank, world_size)

    if (rank == 0):
        # wandb.login(force=True)
        wandb.init(project="diffore", entity="visriv", group='DDP')

    # Load configuration from YAML file
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
    epochs = config.training.optim.max_epochs



    curr_epoch = 0

    model = get_model(config).to(rank)

    if start_from_checkpoint:
        # load weights from checkpoint
        loaded = torch.load(checkpoint_path, 
                            map_location=torch.device('cpu'))
        new_state_dict = {k.replace('module.', ''): v for k, v in loaded['stateDictDecoder'].items()}
        loaded['stateDictDecoder'] = new_state_dict
        # print(loaded['stateDictDecoder'].keys())
        model.load_state_dict(loaded["stateDictDecoder"],
                               strict=False)
        
        get_epoch = re.search(r"model_checkpoint_(\d+)\.pth", checkpoint_path)
        # Extract the number if it exists
        if get_epoch:
            curr_epoch = int(get_epoch.group(1))
        else:
            curr_epoch = 0

    model.train()
    
    # TODO
    train_loader, train_set = get_dataloader(batch_size, rank, world_size, config)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # print model info and trainable weights
    params_trainable = sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])
    params = sum([np.prod(p.size()) for p in model.parameters()])
    #print(model)
    print("Trainable Weights (All Weights): %d (%d)" % (params_trainable, params))



    if config.experiment.train:
        # training loop
        print("\nStarting training...")
        torch_optimizer_class = torchOptimizerClass(config, model, world_size, num_samples=len(train_set))
        optimizer = torch_optimizer_class.get_optimizer()
        scheduler = torch_optimizer_class.get_scheduler()

        
        

        for epoch in range(curr_epoch, epochs):
            train_loader.sampler.set_epoch(epoch)

            losses = []
            with tqdm(train_loader, desc=f"Epoch {epoch}", position=0, leave=True) as pbar:
                for s, sample in enumerate(pbar):

                    optimizer.zero_grad()
                    sample = rearrange(sample, 'b t h w c -> b t c h w').to(rank)
                    d = sample.to(rank)

                    input_steps = config.model.input_steps
                    cond = []
                    for i in range(input_steps):
                        cond += [d[:,i:i+1]] # collect input steps
                    conditioning = torch.concat(cond, dim=2) # combine along channel dimension
                    data = d[:, input_steps:input_steps+1]

                    noise, predicted_x, x, loss_weight, _ = model(conditioning=conditioning, data=data)
                    # loss = F.smooth_l1_loss(x, predicted_x, reduction='none')
                    # loss = reduce(loss, 'b ... -> b (...)', 'mean')
                    # loss = loss * loss_weight#.reshape(loss.shape)
                    # loss = loss.mean()
                    loss = F.smooth_l1_loss(x, predicted_x)

                    pbar.set_postfix({'Batch': s, 'Loss': f'{loss.detach().cpu().item():.7f}'})

                    loss.backward()

                    losses += [loss.detach().cpu().item()]

                    optimizer.step()
            scheduler.step()
            print("[Epoch %2d, FULL]: %1.7f" % (epoch, sum(losses)/len(losses)))
            if(rank==0):
                wandb.log({"train_loss": sum(losses)/len(losses), "epoch": epoch})
            
                # Specify the path to save the checkpoint
                if ((epoch+1) % save_ckpt_every_n_epochs == 0):
                    file_path = os.path.join(save_dir, 'model_checkpoint_{:02d}.pth'.format( epoch))
                    save_checkpoint(model, optimizer, epoch=epoch, file_path=file_path)

                if ((epoch+1) % val_every_n_epochs == 0):
                    validation_step(model, config, rank, save_dir, epoch)
                    torch.cuda.empty_cache()  # Clear the cache after validation


    if config.experiment.validate:
        pred, gt = validation_step(model, config, rank, save_dir, epoch)
   
    cleanup()





if __name__ == "__main__":
    world_size = torch.cuda.device_count()

    parser = argparse.ArgumentParser(description="Distributed Training with PyTorch")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file")
    
    args = parser.parse_args()

    # Load the configuration from the provided path
    config = OmegaConf.load(args.config)
        
    main_child(rank=0, world_size=1, config=config)
    # mp.spawn(main_child,
    #          args=(world_size, config),
    #          nprocs=world_size,
            #  join=True)
