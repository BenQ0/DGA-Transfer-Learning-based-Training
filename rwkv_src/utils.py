import torch
from torch.utils.data import Dataset

class RWKV_Dataset(Dataset):
    def __init__(self, domains_int, labels, ctx_len, vocab_size,device="cuda"):
        assert len(domains_int) == len(labels)
        self.x = torch.tensor(domains_int, dtype=torch.long)
        self.y = labels

        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        self.device = device

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        # y = torch.tensor([self.y[index]] * len(x), dtype=torch.long,
        #                     device=torch.device('cuda'))
        return x,y