import os
import sys
from tqdm import tqdm
project_root = '/home/users/nus/e1333861/decodiff'
sys.path.append(project_root)
from omegaconf import OmegaConf
from src.model.diffusion import * #DiffusionModel  # Use absolute imports
from src.kol_dataloader import kolTorchDataset
from src.utils.get_model import get_model

from torch.utils.data import DataLoader, DistributedSampler, Dataset
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = OmegaConf.load('/home/users/nus/e1333861/decodiff/configs/diffusion_denoise.yaml')



model = get_model(config)

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


# data = torch.rand(2, 1, 18, 128, 128)
# cond = torch.rand(2, 1, 2, 128, 128)


# train_loader, train_set = get_dataloader(2, 0, 1, config, "train")
val_loader = get_validation_dataloader(2, 0, 1, config)


rank = 0
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
numSamples = 1
# sampling loop
print("\nStarting sampling...")
gt = []
pred = []

model.eval()

for s, sample in enumerate(val_loader, 0):
    sample = rearrange(sample, 'b t h w c -> b t c h w').to(device)
    d = sample.to(device).repeat(numSamples,1,1,1,1) # reuse batch dim for samples

    prediction = torch.zeros_like(d, device=device)
    inputSteps = config.model.input_steps

    for i in range(inputSteps): # no prediction of first steps
        prediction[:,i] = d[:,i]

    for i in tqdm(range(inputSteps, d.shape[1]), desc="autoreg rollout steps"):
        # print('{} steps out of'.format(i), d.shape[1])
        cond = []
        for j in range(inputSteps,0,-1):
            cond += [prediction[:, i-j : i-(j-1)]] # collect input steps
        cond = torch.concat(cond, dim=2) # combine along channel dimension

        result, interm_features = model(conditioning=cond, data=d[:,i-1:i]) # auto-regressive inference
        
        prediction[:,i:i+1] = result.to(device)




# with torch.no_grad():
#     for s, sample in enumerate(val_loader, 0):
#         if (s == config.validation.n_val_samples):
#             break
#         sample = rearrange(sample, 'b t h w c -> b t c h w').to(device)
#         print('sample shape from testLoader:', sample.unsqueeze(0).cpu().numpy().shape)
#         gt += [sample.unsqueeze(0).cpu().numpy()]
#         d = sample.to(device).repeat(numSamples,1,1,1,1) # reuse batch dim for samples
#         print('d shape from testLoader:', d.shape)

#         prediction = torch.zeros_like(d, device=device)
#         inputSteps = config.model.input_steps

#         for i in range(inputSteps): # no prediction of first steps
#             prediction[:,i] = d[:,i]

#         for i in tqdm(range(inputSteps, d.shape[1]), desc="autoreg rollout steps"):
#             # print('{} steps out of'.format(i), d.shape[1])
#             cond = []
#             for j in range(inputSteps,0,-1):
#                 cond += [prediction[:, i-j : i-(j-1)]] # collect input steps
#             cond = torch.concat(cond, dim=2) # combine along channel dimension

#             result, interm_features = model(conditioning=cond, data=d[:,i-1:i]) # auto-regressive inference
            
#             prediction[:,i:i+1] = result.to(device)

#         prediction = torch.reshape(prediction, (numSamples, -1, d.shape[1], d.shape[2], d.shape[3], d.shape[4]))
#         pred += [prediction.cpu().numpy()]
#         print("  Sequence %d finished" % s)
        
#         del prediction, sample, d, result, cond
#         torch.cuda.empty_cache()

# # print(output.shape)

