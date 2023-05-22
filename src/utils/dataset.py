import pandas as pd
import torch
from torch.utils.data import Dataset


class MultiViewDataset(Dataset):
    def __init__(self, Xs):
        self.data = Xs
        self.samples = pd.Index(set(sum([X.index.to_list() for X in Xs], [])))


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        sample = self.samples[idx]
        instance = [torch.Tensor(X.loc[sample].values) for X in self.data if sample in X.index]
        return instance

