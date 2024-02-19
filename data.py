from torch.utils.data import DataLoader
import numpy as np
import torch

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


class BatchLoader(torch.utils.data.Dataset):
    def __init__(self, er_vocab, num_e):
        self.num_e = num_e
        sub_rel_idx = torch.Tensor(list(er_vocab.keys())).long()
        self.sub_idx = sub_rel_idx[:, 0]
        self.rel_idx = sub_rel_idx[:, 1]
        self.obj_idx = list(er_vocab.values())
        assert len(self.sub_idx) == len(self.rel_idx) == len(self.obj_idx)

    def __len__(self):
        return len(self.obj_idx)

    def __getitem__(self, idx):
        y = torch.zeros(self.num_e)
        y[self.obj_idx[idx]] = 1  # set 1's for all objects
        return self.sub_idx[idx], self.rel_idx[idx], y
