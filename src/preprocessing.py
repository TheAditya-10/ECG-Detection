import os
import numpy as np
import wfdb
from scipy.signal import resample
import torch
from torch.utils.data import Dataset, DataLoader

# ----- CONFIG -----
DATA_DIR = "data/mitdb"
SEQ_LEN = 500  # Fixed sequence length for model

class ECGDataset(Dataset):
    def __init__(self, data_dir):
        self.records = [f[:-4] for f in os.listdir(data_dir) if f.endswith('.dat')]
        self.data_dir = data_dir

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        signals, _ = wfdb.rdsamp(os.path.join(self.data_dir, record))
        signals = resample(signals[:, 0], SEQ_LEN)  # Resample to fixed length
        signals = (signals - np.mean(signals)) / np.std(signals)  # Normalize
        label = 1 if "abnormal" in record else 0  # Dummy binary label
        return torch.tensor(signals, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def get_dataloaders(batch_size=32, train_split=0.8):
    dataset = ECGDataset(DATA_DIR)
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader
