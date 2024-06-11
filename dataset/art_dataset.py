import numpy as np
from torch.utils.data import Dataset
import torch

class ArtDataset(Dataset):
    def __init__(self, data):
        to1hot = np.eye(2)
        self.dataset = []
        for d, label in data:
            self.dataset += [(im, to1hot[label]) for im in d]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        vec, label = self.dataset[index]
        return torch.tensor(vec, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
