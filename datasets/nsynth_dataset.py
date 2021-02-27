import os.path as osp
import pandas as pd
import librosa as lr
import torch

from torch.utils.data import Dataset


# generic sofa dataset
class NsynthDataset(Dataset):
    def __init__(self, dataset_path, transform=None, 
                 sr=None, duration=2,
                 pitches=None, velocities=None, instrument_sources=None, instrument_families=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.sr = sr
        self.duration = duration
        self.pitches = pitches
        self.velocities = velocities
        self.instrument_sources = instrument_sources
        self.instrument_families = instrument_families
        self.df = None
        self.load_data()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx: int):
        item = self.df.iloc[idx]
        filepath = osp.join(self.dataset_path, 'audio', f'{item.name}.wav')
        sample, sr = lr.load(filepath, sr=self.sr, duration=self.duration, mono=True)
        sample = lr.util.fix_length(sample, sr * self.duration)
        sample = torch.tensor(sample)
        if self.transform:
            sample = self.transform(sample)
        item = item.drop('qualities_str').to_dict()
        return sample, item

    def load_data(self):
        filepath_cache = osp.join(self.dataset_path, 'examples_cache.pkl')
        if osp.exists(filepath_cache):
            #print(f'Loading cached data: {filepath_cache}')
            _df = pd.read_pickle(filepath_cache)
        else:
            filepath = osp.join(self.dataset_path, 'examples.json')
            #print(f'Caching data: {filepath}')
            _df = pd.read_json(filepath).T
            _df.to_pickle(filepath_cache)
        # filter data
        if self.pitches:
            _df = _df[_df['pitch'].isin(self.pitches)]
        if self.velocities:
            _df = _df[_df['velocity'].isin(self.velocities)]
        if self.instrument_sources:
            _df = _df[_df['instrument_source'].isin(self.instrument_sources)]
        if self.instrument_families:
            _df = _df[_df['instrument_family'].isin(self.instrument_families)]
        self.df = _df
        #print(f'Data: {_df.shape}')
