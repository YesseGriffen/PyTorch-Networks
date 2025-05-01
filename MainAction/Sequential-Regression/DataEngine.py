import torch
from torch.utils.data import Dataset, BatchSampler
from collections import deque

#---------------------------------------------
# Create Sliding Window Dataset
# Pairs of X, Y
# Buffer of last values
# Function to append new data 1 by 1
class TSD(Dataset):
    def __init__(self, series: torch.Tensor, data_params: list):
        # Create X, Y pair through raw sequence, and buffer of last points
        self.window, self.features, self.step, self.forecast, self.batch_size = data_params
        tot_window = (self.window + self.forecast)
        sequence        = series.unfold(0, tot_window, self.step)
        self.X          = sequence[:, :self.window].unsqueeze(-1)
        self.Y          = sequence[:, self.window:]
        self.buffer     = deque(series[-tot_window:].tolist(), maxlen=tot_window)
        self.counter    = 0
        
        self.max_samples = 500
        
    def append(self, value: float):
        # Append new value to rolling-buffer
        self.buffer.append(value)
        self.counter += 1
        # Add new window if able
        if self.counter >= self.step:
            data = torch.Tensor(self.buffer)
            X = data[:self.window].unsqueeze(0).unsqueeze(-1)
            Y = data[self.window:].unsqueeze(0)
            self.X = torch.cat([self.X, X], dim=0)[-self.max_samples:]
            self.Y = torch.cat([self.Y, Y], dim=0)[-self.max_samples:]
            self.counter = 0

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

#---------------------------------------------
# BatchSampler used for negating drop_last, and instead "drop_first"
# So we keep more relevant data, rather than dropping it. 
# But we drop the data points at the front if they dont fit into a batch_size
class TailS(BatchSampler):
    def __init__(self, dataset: Dataset, batch_size: int, batch_range: int = 0):
        # Grabs remainder of X Y pairs that cannot be put into a batch
        total_len = len(dataset)
        rem = total_len % batch_size
        idxs = list(range(rem, total_len))
        # Grabs batches based on remainder
        self.batches = [idxs[i : i + batch_size] for i in range(0, len(idxs), batch_size)]
        if batch_range != 0:
            self.batches = self.batches[-batch_range:]

        super().__init__(idxs, batch_size, drop_last=False)

    def __iter__(self):
        return iter(self.batches)
    
    def __len__(self):
        return len(self.batches)
    
    def __getitem__(self, idx):
        # Support single index and slicing
        return self.batches[idx]
