"""Epochs written by make_epochs.py, split by participant into training, validation and test sets for one fold.

Adapted from the dataset loaders of CBraMod (https://github.com/wjq-learning/CBraMod, MIT licence).
"""
import os
import pickle
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

EPOCH_ID = re.compile(r'^(sub-\d+)_task-(walkstraight|walkcurved|headshake|headnod|pickup|stairs)'
                      r'_acq-(slow|fast)move_epoch\d+$')


def epoch_id_of(path):
    return os.path.splitext(os.path.basename(path))[0]


class CustomDataset(Dataset):
    """Epochs of the given files with channel_size channels; an item is (signal / 100, label, epoch_id)."""

    def __init__(self, files, mode, channel_size=20, window_size=1):
        super(CustomDataset, self).__init__()
        self.channel_size = channel_size
        self.window_size = window_size
        self.files = []
        for path in files:
            with open(path, 'rb') as f:
                if pickle.load(f)['signal'].shape[0] == channel_size:
                    self.files.append(path)
        if len(self.files) < len(files):
            print(f'[{mode}] left out {len(files) - len(self.files)} epochs with a channel count other than {channel_size}')
        self.epoch_ids = [epoch_id_of(path) for path in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], 'rb') as f:
            sample = pickle.load(f)
        data = sample['signal']
        if data.size != self.channel_size * self.window_size * 200:
            raise ValueError(f'{self.files[idx]}: signal of shape {data.shape}, expected '
                             f'{self.channel_size} channels x {self.window_size} x 200 samples')
        data = data.reshape(self.channel_size, self.window_size, 200)
        return data / 100.0, sample['label'], self.epoch_ids[idx]

    def collate(self, batch):
        x = np.array([item[0] for item in batch])
        y = np.array([item[1] for item in batch])
        return torch.from_numpy(x).float(), torch.from_numpy(y).long(), [item[2] for item in batch]


class LoadDataset(object):
    def __init__(self, params):
        self.params = params

    def get_data_loader(self):
        """Test and validation sets: one participant each (params.test_subject, params.val_subject); training set:
        all other participants."""
        folder = self.params.datasets_dir
        by_subject = {}
        for name in sorted(os.listdir(folder)):
            if not name.endswith('.pickle'):
                continue
            m = EPOCH_ID.match(epoch_id_of(name))
            if m is None:
                raise ValueError(f'{name}: not an epoch name of make_epochs.py')
            by_subject.setdefault(m.group(1), []).append(os.path.join(folder, name))
        subjects = sorted(by_subject)
        test, val = self.params.test_subject, self.params.val_subject
        if test not in by_subject or val not in by_subject:
            raise ValueError(f'test_subject={test} and val_subject={val} must be among {subjects}')
        if test == val:
            raise ValueError('test_subject and val_subject must be different')
        train = [s for s in subjects if s not in (test, val)]
        print(f'{len(subjects)} participants; train ({len(train)}): {train}; val: {val}; test: {test}')

        sets = {
            'train': CustomDataset([f for s in train for f in by_subject[s]], 'train',
                                   self.params.channel_size, self.params.window_size),
            'val': CustomDataset(by_subject[val], 'val', self.params.channel_size, self.params.window_size),
            'test': CustomDataset(by_subject[test], 'test', self.params.channel_size, self.params.window_size),
        }
        print(f"Epochs - train: {len(sets['train'])}, val: {len(sets['val'])}, test: {len(sets['test'])}")
        return {mode: DataLoader(ds, batch_size=self.params.batch_size, collate_fn=ds.collate, shuffle=mode == 'train')
                for mode, ds in sets.items()}
