import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import torch
import os

class kol_datamodule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # Download or prepare data here if required
        # Example: torchvision.datasets.MNIST(self.data_dir, train=True, download=True)
        pass

    def setup(self, stage: str = None):
        # Load data and split datasets here
        # Example: Train/Validation split
        dataset = torch.utils.data.TensorDataset(torch.randn(1000, 10), torch.randint(0, 2, (1000,)))
        if stage == 'fit' or stage is None:
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

        if stage == 'test' or stage is None:
            self.test_dataset = dataset  # For simplicity, use the same dataset for testing

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

