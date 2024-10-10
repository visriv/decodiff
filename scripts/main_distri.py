import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import wandb

from network import *
from diffusion import *
from dataloader import *
from kol_dataloader import kolTorchDataset
from validation_distri import validation_step
from omegaconf import OmegaConf
import wandb


os.environ['MASTER_ADDR'] = 'localhost' 
os.environ['MASTER_PORT'] = '12355'   
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
    return dataloader





def train(rank, world_size, config):
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
    if start_from_checkpoint:
        epochs = config.training.finetune_epochs
        lr = config.training.finetune_learning_rate
    else:
        epochs = config.training.epochs
        lr = config.training.learning_rate




    model = DiffusionModel(config).to(rank)

    if start_from_checkpoint:
        # load weights from checkpoint
        loaded = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(loaded["stateDictDecoder"])
    model.train()
    
    # TODO
    train_loader = get_dataloader(batch_size, rank, world_size, config)
    model = DDP(model, device_ids=[rank])

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
            train_loader.sampler.set_epoch(epoch)

            losses = []
            for s, sample in enumerate(train_loader, 0):
                optimizer.zero_grad()
                print(s)
                sample = rearrange(sample, 'b t h w c -> b t c h w').to(rank)
                d = sample.to(rank)

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
            if(rank==0):
                wandb.log({"train_loss": sum(losses)/len(losses), "epoch": epoch})
            
                # Specify the path to save the checkpoint
                if (epoch % save_ckpt_every_n_epochs == 0):
                    file_path = os.path.join(save_dir, 'model_checkpoint_{:02d}.pth'.format( epoch))
                    save_checkpoint(model, optimizer, epoch=epoch, file_path=file_path)

                if ((epoch+1) % val_every_n_epochs == 0):
                    validation_step(model, config, rank, save_dir, epoch)
                    torch.cuda.empty_cache()  # Clear the cache after validation



   
    cleanup()





if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    config = OmegaConf.load('/home/users/nus/e1333861/autoreg-pde-diffusion/scripts/kol.yaml')
    
    mp.spawn(train,
             args=(world_size, config),
             nprocs=world_size,
             join=True)
