import os
import numpy as np
import wfdb
from scipy.signal import resample
import torch
from torch.utils.data import Dataset, DataLoader

DATA_DIR = "data/mitdb"
SEQ_LEN = 360  
TRANSFORM_LEN = 12 # Fixed sequence length for transformer input -> Why 360 ?

class ECGDataset(Dataset):
    def __init__(self, data_dir, seq_len=SEQ_LEN, transform_len = TRANSFORM_LEN, augment=False):
        self.records = [f[:-4] for f in os.listdir(data_dir) if f.endswith('.dat')]  # Get record names as list of strings without .dat
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.augment = augment
        self.transform_len = transform_len

    def __len__(self):
        return len(self.records) * (648000 // self.seq_len)  # Total number of 1-second segments across all records

    def __getitem__(self, idx):
        # Determine which record and segment to load
        record_idx = idx // (648000 // self.seq_len)  # Record index
        segment_idx = idx % (648000 // self.seq_len)  # Segment index within the record

        record = self.records[record_idx]
        try:
            signals, _ = wfdb.rdsamp(os.path.join(self.data_dir, record))  # Read the full signal
        except Exception as e:
            raise RuntimeError(f"Error reading record {record}: {e}")

        # Use only the first lead
        signals = signals[:, 1]

        # Extract the 1-second segment
        start = segment_idx * self.seq_len
        end = start + self.seq_len
        segment = signals[start:end]

        # Resample the segment to 120 Hz
        segment = resample(segment, self.transform_len)

        # Normalize the segment
        segment = (segment - np.mean(segment)) / np.std(segment)

        # Optional data augmentation
        if self.augment:
            segment = self._augment_signal(segment)

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
            torch.tensor(segment, dtype=torch.float32),
            torch.tensor(anomaly_label, dtype=torch.long),
            torch.tensor(class_label, dtype=torch.long),
        )

    def _augment_signal(self, segment):
        """Apply random noise or scaling for data augmentation."""
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.01, size=segment.shape)
            segment += noise
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            segment *= scale
        return segment

def get_dataloaders(batch_size=32, train_split=0.8, augment=False):
    dataset = ECGDataset(DATA_DIR, augment=augment)
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))  # Randomly split dataset with a fixed seed for reproducibility
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader
