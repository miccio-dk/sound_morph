import os.path as osp
import pandas as pd
import librosa as lr
import torch

from torch.utils.data import Dataset


# generic sofa dataset
class UrbanSoundsDataset(Dataset):
    def __init__(self, dataset_path, transform=None, 
                 sr=None, duration=2, classes=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.sr = sr
        self.duration = duration
        self.classes = classes
        self.df = None
        self.load_data()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx: int):
        item = self.df.iloc[idx]
        filepath = osp.join(self.dataset_path, f'{item.name}.wav')
        sample, sr = lr.load(filepath, sr=self.sr, duration=self.duration, mono=True)
        sample = lr.util.fix_length(sample, int(sr * self.duration))
        sample = torch.tensor(sample)
        if self.transform:
            sample = self.transform(sample)
        return sample, item.to_dict()

    def load_data(self):
        filepath_cache = osp.join(self.dataset_path, 'train.pkl')
        if osp.exists(filepath_cache):
            #print(f'Loading cached data: {filepath_cache}')
            _df = pd.read_pickle(filepath_cache)
        else:
            filepath = osp.join(self.dataset_path, 'train.csv')
            #print(f'Caching data: {filepath}')
            _df = pd.read_csv(filepath, index_col='ID')
            _df.to_pickle(filepath_cache)
        # filter data
        if self.classes:
            _df = _df[_df['Class'].isin(self.classes)]
        self.df = _df
        #print(f'Data: {_df.shape}')
