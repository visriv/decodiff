import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

def warmup_lambda(warmup_steps, min_lr_ratio=0.1):
    def ret_lambda(epoch):
        if epoch <= warmup_steps:
            return min_lr_ratio + (1.0 - min_lr_ratio) * epoch / warmup_steps
        else:
            return 1.0
    return ret_lambda


class torchOptimizerClass():
    def __init__(self, config, model, world_size, num_samples):


        total_batch_size = total_batch_size = int(config.training.batch_size  * world_size)
        self.total_num_steps = int(config.training.optim.max_epochs * num_samples / total_batch_size)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),  # Model parameters
            lr=config.training.optim.lr,             # Learning rate
            betas=(config.training.optim.betas[0], config.training.optim.betas[1]),  # Coefficients used for computing running averages of gradient and its square
            eps=1e-8,            # Term added to improve numerical stability
            weight_decay=config.training.optim.wd,   # Weight decay (L2 regularization)
            amsgrad=False        # Whether to use the AMSGrad variant of the algorithm
        )
        warmup_iter = int(np.round(config.training.optim.warmup_percentage * self.total_num_steps))
        warmup_scheduler = LambdaLR(self.optimizer,
                                    lr_lambda=warmup_lambda(warmup_steps=warmup_iter,
                                                            min_lr_ratio=config.training.optim.warmup_min_lr_ratio))
        cosine_scheduler = CosineAnnealingLR(self.optimizer,
                                                T_max=(self.total_num_steps - warmup_iter),
                                                eta_min=config.training.optim.min_lr_ratio * config.training.optim.lr)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                    milestones=[warmup_iter])


    def get_optimizer(self):
        return self.optimizer

    def get_scheduler(self):
        return self.scheduler

