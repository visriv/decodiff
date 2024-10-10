import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy


import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

class MyLightningModule(LatentDiffusion):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', val_loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)




# Instantiate DataModule
data_module = MyDataModule(data_dir='path/to/data', batch_size=32, num_workers=4)

# Instantiate LightningModule
model = MyLightningModule(input_dim=10, hidden_dim=64, output_dim=2)

# Trainer setup with DDP strategy for distributed training
trainer = pl.Trainer(
    max_epochs=10,
    accelerator='gpu',  # Use GPU if available
    devices=4,  # Number of GPUs to use
    strategy=DDPStrategy(find_unused_parameters=False),  # Setup for DDP training
    precision=16,  # Mixed precision training, optional
)

# Train the model
trainer.fit(model, data_module)

# Validate the model
trainer.validate(model, data_module)
