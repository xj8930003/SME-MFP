import torch
from torch.utils.data import DataLoader
import numpy as np

class Dataset(object):
    def __init__(self, data):
        self.train_values = data[0]
        self.train_label = data[1]

    def __getitem__(self, item):
        value = torch.from_numpy(self.train_values[item]).type(torch.int)
        label = torch.from_numpy(self.train_label[item]).type(torch.float)
        return (value, label )

    def __len__(self):
        return self.train_values.shape[0]

if __name__ == '__main__':
    v = np.random.rand(5,4)
    l = np.random.randint(0,2,(5,2))
    D = [v,l]
    Mydata = Dataset(D)
    for t,b in Mydata:
        print(t)
        print(b)
        assert 0
