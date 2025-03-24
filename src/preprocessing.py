import os
import numpy as np
import wfdb
from scipy.signal import resample
import torch
from torch.utils.data import Dataset, DataLoader

DATA_DIR = "data/mitdb"
SEQ_LEN = 360  # Fixed sequence length for transformer input -> Why 360 ?

class ECGDataset(Dataset):
    def __init__(self, data_dir, seq_len=SEQ_LEN, augment=False):
        self.records = [f[:-4] for f in os.listdir(data_dir) if f.endswith('.dat')]     # Get record names as list of String without .dat
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        try:
            signals, _ = wfdb.rdsamp(os.path.join(self.data_dir, record))       # The signal is returned as a 2D array (channels × samples)
        except Exception as e:
            raise RuntimeError(f"Error reading record {record}: {e}")


        # # Modifying Steps (CBC)

        # Use only the first lead
        signals = signals[:, 0]

        # Resample to fixed length
        signals = resample(signals, self.seq_len)

        # Normalize signals
        signals = (signals - np.mean(signals)) / np.std(signals)

        # Optional data augmentation
        if self.augment:
            signals = self._augment_signal(signals)



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

    def _augment_signal(self, signal):
        """Apply random noise or scaling for data augmentation."""
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.01, size=signal.shape)
            signal += noise
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            signal *= scale
        return signal

def get_dataloaders(batch_size=32, train_split=0.8, augment=False):
    dataset = ECGDataset(DATA_DIR, augment=augment)
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])       # Randomly split dataset into non-overlapping new datasets of given lengths (CBC)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader
