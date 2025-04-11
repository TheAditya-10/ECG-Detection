import os
import numpy as np
import wfdb
from scipy.signal import resample
import torch
from torch.utils.data import Dataset, DataLoader

DATA_DIR = "data/mitdb"
SEQ_LEN = 360
TRANSFORM_LEN = 1  # Fixed sequence length for transformer input

class ECGDataset(Dataset):
    def __init__(self, records, data_dir, seq_len=SEQ_LEN, transform_len=TRANSFORM_LEN, augment=False, pipeline=None):
        self.records = records  # List of record names
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.augment = augment
        self.transform_len = transform_len
        self.pipeline = pipeline  # Preprocessing pipeline function

    def __len__(self):
        return len(self.records) * (650000 // self.seq_len)  # Total number of 1-second segments across all records

    def __getitem__(self, idx):
        # Determine which record and segment to load
        record_idx = idx // (650000 // self.seq_len)  # Record index
        segment_idx = idx % (650000 // self.seq_len)  # Segment index within the record

        record = self.records[record_idx]
        try:
            signals, _ = wfdb.rdsamp(os.path.join(self.data_dir, record))  # Read the full signal
            annotation = wfdb.rdann(os.path.join(self.data_dir, record), "atr")  # Read the annotation file
        except Exception as e:
            raise RuntimeError(f"Error reading record {record}: {e}")

        # Use only the first lead
        signals = signals[:, 0]

        # Extract the 1-second segment
        start = segment_idx * self.seq_len
        end = start + self.seq_len
        segment = signals[start:end]

        # Print the segment for debugging
        # print(segment)

        # Fetch labels for the segment
        anomaly_label, class_label = self.get_labels(annotation, start, end)

        # Apply the preprocessing pipeline
        if self.pipeline:
            segment = self.pipeline(segment, self.augment)

        return (
            torch.tensor(segment, dtype=torch.float32),
            torch.tensor(anomaly_label, dtype=torch.long),
            torch.tensor(class_label, dtype=torch.long),
        )

    def get_labels(self, annotation, start, end):
        """Fetch labels for a given segment based on annotations."""
        # Define a mapping for annotation symbols to labels
        annotation_mapping = {
            "N": (0, 0),  # Normal beat
            "L": (1, 1),  # Left bundle branch block beat
            "R": (1, 2),  # Right bundle branch block beat
            "A": (1, 3),  # Atrial premature contraction
            "a": (1, 4),  # Aberrated atrial premature contraction
            "V": (1, 5),  # Premature ventricular contraction
            "F": (1, 6),  # Fusion of ventricular and normal beat
            "J": (1, 7),  # Nodal (junctional) premature beat
            "E": (1, 8),  # Ventricular escape beat
            "/": (1, 9),  # Paced beat
            "Q": (1, 10),  # Unclassifiable beat
            "?": (1, 11),  # Unclassifiable beat
            "(N": (0, 0),  # Normal sinus rhythm
            "(AFIB": (1, 12),  # Atrial fibrillation
            "(AFL": (1, 13),  # Atrial flutter
            "(SVTA": (1, 14),  # Supraventricular tachyarrhythmia
            "(VT": (1, 15),  # Ventricular tachycardia
            "(IVR": (1, 16),  # Idioventricular rhythm
            "(VFL": (1, 17),  # Ventricular flutter
            "(PACE": (1, 18),  # Pacemaker rhythm
            "(B": (1, 19),  # Bigeminy
            "(T": (1, 20),  # Trigeminy
        }

        # Initialize labels for the segment
        anomaly_label = 0  # Default to normal
        class_label = 0

        # Find annotations within the segment
        for sample, symbol in zip(annotation.sample, annotation.symbol):
            if start <= sample < end:
                # Map the annotation symbol to labels
                mapped_label = annotation_mapping.get(symbol, (0,0))  # Default to anomaly if unknown
                anomaly_label = max(anomaly_label, mapped_label[0])  # If any anomaly is found, set anomaly_label to 1
                class_label = max(class_label, mapped_label[1])  # Use the most severe class label

        return anomaly_label, class_label

def preprocessing_pipeline(segment, augment=False):
    """Pipeline for preprocessing ECG segments."""
    # Resample the segment to 120 Hz
    segment = resample(segment, TRANSFORM_LEN)

    # Normalize the segment
    # epsilon = 1e-8  # Small value to prevent division by zero
    # segment = (segment - np.mean(segment)) / (np.std(segment))

    # Apply data augmentation if enabled
    if augment:
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.01, size=segment.shape)
            segment += noise
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            segment *= scale

    return segment

def get_dataloaders(batch_size=32, train_split=0.8, augment=False):
    # Get all record names
    all_records = [f[:-4] for f in os.listdir(DATA_DIR) if f.endswith('.dat')]

    # Split records into training and test sets
    train_size = int(train_split * len(all_records))
    train_records = all_records[:train_size]
    test_records = all_records[train_size:]

    # Create datasets with the preprocessing pipeline
    train_dataset = ECGDataset(train_records, DATA_DIR, augment=augment, pipeline=preprocessing_pipeline)
    test_dataset = ECGDataset(test_records, DATA_DIR, augment=False, pipeline=preprocessing_pipeline)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader
