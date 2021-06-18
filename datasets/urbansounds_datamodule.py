import os.path as osp

from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
from torchvision.transforms import Compose
from .urbansounds_dataset import UrbanSoundsDataset
from .transforms import Stft


class UrbanSoundsModule(LightningDataModule):
    def __init__(self, configs, num_workers=4, batch_size=32):
        super().__init__()
        # store params
        self.configs = configs
        self.num_workers = num_workers
        self.batch_size = batch_size
        # setup transforms
        if self.configs['feature'] == 'spec':
            #print(self.configs['feature_params'])
            self.transform = Stft(n_fft=self.configs['n_fft'], **self.configs['feature_params'])
        ## TODO implement more features?

    def setup(self, stage=None):
        # train/val
        if stage == 'fit' or stage is None:
            self.data_all = UrbanSoundsDataset(self.configs['dataset_path'], transform=self.transform, **self.configs['ds_kwargs'])
            # split data
            train_len = int(len(self.data_all) * 0.8)
            val_len = len(self.data_all) - train_len
            self.data_train, self.data_val = random_split(self.data_all, [train_len, val_len])
            self.dims = self.data_train[0][0].shape
        # test
        if stage == 'test' or stage is None:
            pass

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return None
