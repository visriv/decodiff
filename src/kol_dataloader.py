"""
Code is adapted from https://github.com/amazon-science/earth-forecasting-transformer/blob/e60ff41c7ad806277edc2a14a7a9f45585997bd7/src/earthformer/datasets/sevir/sevir_torch_wrap.py
Add data augmentation.
Only return "VIL" data in `torch.Tensor` format instead of `Dict`
"""

import h5py
import torch
from torch.utils.data import Dataset as TorchDataset






class kolTorchDataset(TorchDataset):

    def __init__(self, data_path='datasets/kol/results.h5', 
                 split="train", 
                 window_length = 1,
                 k=1, # number to samples to skip (1/(k))
                 train_ratio = 0.8, 
                 val_ratio = 0.1, 
                 standardize=True,
                 flatten = False, crop=0):
        super().__init__()
        # === parameters ===
        self.standardize = standardize
        self.crop = crop
        self.window_length = window_length
        # === data preprocess ===
        # print(os.getcwd())
        self.data_file = data_path
        hdf_file = h5py.File(self.data_file, 'r')
        velocity = hdf_file['velocity_field'][:]

        self.data = velocity
        self.row = velocity.shape[1]
        self.col = velocity.shape[2]
        self.num_channels = velocity.shape[3]

        self.k = k



        

        # === dataset split ===
        assert split in ("train", "val", "test"), "Unknown dataset split"
        assert train_ratio + val_ratio < 1, "train_ratio + val_ratio must be less than 1"

        # crop if required
        if (crop > 0):
            min_y = self.row//2-int(crop/2)
            max_y = self.row//2+int(crop/2)
            min_x = self.col//2-int(crop/2)
            max_x = self.col//2+int(crop/2)
            self.data = self.data[:, min_y:max_y, min_x:max_x]

        self.num_rows = self.data.shape[1]
        self.num_cols = self.data.shape[2]


        # flatten spatial dimensions
        if (flatten == True):
            self.data = self.data.reshape(self.data.shape[0], -1) # (n_t, N_grid)
        
        # self.data = self.data[...,None] # (n_t, N_grid, d_features)
        self.nt = self.data.shape[0]

        self.train_nt = int(self.nt * train_ratio)
        self.val_nt = int(self.nt * val_ratio)
        self.test_nt = self.nt - self.train_nt - self.val_nt

        # === normalization of sample ===
        # calculate mean and std using training data
        if self.standardize:
            if (flatten):
                self.data_mean = self.data[:self.train_nt].mean(axis=0)
                self.data_std = self.data[:self.train_nt].std(axis=0)
            else:
                self.data_mean = self.data[:self.train_nt].mean()
                self.data_std = self.data[:self.train_nt].std()
            self.data = (self.data - self.data_mean)/self.data_std

        if split == "train":
            self.data = self.data[:self.train_nt]
        elif split == "val":
            self.data = self.data[self.train_nt:self.train_nt+self.val_nt]
        elif split == "test":
            self.data = self.data[self.train_nt+self.val_nt:]

            
        self.nt = self.data.shape[0]


        # === create samples by sliding windows ===
        self.n_samples = self.nt-self.window_length+1 # number of samples in dataset

    def __len__(self):
        # return 100
        return self.n_samples

    def __getitem__(self, idx):
        # === Sample ===
        sampled_array = self.data[::self.k, :, :, :]
        idx = idx//self.k
        sample = sampled_array[idx:idx+self.window_length]


        sample = sample.reshape(-1,
                                self.window_length, 
                                self.num_rows,
                                self.num_cols,
                                self.num_channels) # num_channels
        
 
        sample = torch.tensor(sample, dtype = torch.float32)
        sample = torch.squeeze(sample, 0)
        return sample
    
