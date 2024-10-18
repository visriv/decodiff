import torch
import wandb
import torch.nn.functional as F

def log_error(gt, pred):
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