import os
import numpy as np
import wfdb
from scipy.signal import resample
import torch
from torch.utils.data import Dataset, DataLoader

DATA_DIR = "data/mitdb"
SEQ_LEN = 500  

class ECGDataset(Dataset):
    def __init__(self, data_dir):
        self.records = [f[:-4] for f in os.listdir(data_dir) if f.endswith('.dat')]
        self.data_dir = data_dir

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        signals, _ = wfdb.rdsamp(os.path.join(self.data_dir, record))
        signals = resample(signals[:, 0], SEQ_LEN)
        signals = (signals - np.mean(signals)) / np.std(signals)  

        # Simulated labels - replace with real ones
        if "normal" in record:
            anomaly_label, class_label = 0, 0
        elif "afib" in record:
            anomaly_label, class_label = 1, 1
        elif "vtach" in record:
            anomaly_label, class_label = 1, 2
        elif "pvc" in record:
            anomaly_label, class_label = 1, 3
        else:
            anomaly_label, class_label = 1, 4  

        return (
            torch.tensor(signals, dtype=torch.float32), 
            torch.tensor(anomaly_label, dtype=torch.long),
            torch.tensor(class_label, dtype=torch.long)
        )

def get_dataloaders(batch_size=32, train_split=0.8):
    dataset = ECGDataset(DATA_DIR)
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader
