from torch.utils.data import DataLoader
import numpy as np
import torch

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


class DatasetTriple(torch.utils.data.Dataset):
    def __init__(self, data):
        print(data[:5])
        exit()
        data = torch.Tensor(data).long()
        self.sub_idx = data[:, 0]
        self.rel_idx = data[:, 1]
        self.obj_idx = data[:, 2]

        assert self.sub_idx.shape == self.rel_idx.shape == self.obj_idx.shape

        self.length = len(self.sub_idx)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        s = self.sub_idx[idx]
        r = self.rel_idx[idx]
        o = self.obj_idx[idx]
        return s, r, o


class SubjectAndRelationBatchLoader(torch.utils.data.Dataset):
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
        y_vec = torch.zeros(self.num_e)
        y_vec[self.obj_idx[idx]] = 1  # given subject and rel, set 1's for all objects.
        return self.sub_idx[idx], self.rel_idx[idx], y_vec
