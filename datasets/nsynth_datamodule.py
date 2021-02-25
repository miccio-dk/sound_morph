import os.path as osp

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from torchvision.transforms import Compose
from .nsynth_dataset import NsynthDataset
from .transforms import Stft


class NsynthDataModule(LightningDataModule):
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
            dataset_path_train = osp.join(self.configs['dataset_path'], 'nsynth-train')
            self.data_train = NsynthDataset(dataset_path_train, transform=self.transform, **self.configs['ds_kwargs'])
            dataset_path_val = osp.join(self.configs['dataset_path'], 'nsynth-valid')
            self.data_val = NsynthDataset(dataset_path_val, transform=self.transform, **self.configs['ds_kwargs'])
            self.dims = self.data_train[0][0].shape
        # test
        if stage == 'test' or stage is None:
            dataset_path_test = osp.join(self.configs['dataset_path'], 'nsynth-test')
            self.data_test = NsynthDataset(dataset_path_test, transform=self.transform, **self.configs['ds_kwargs'])
            if len(self.data_test) > 0:
                self.dims = getattr(self, 'dims', self.data_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
