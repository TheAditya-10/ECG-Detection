import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wfdb
from scipy.signal import resample

# ----- CONFIG -----
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.001
SEQ_LEN = 500  # Number of samples in each sequence
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "mitdb"  # Path to MIT-BIH dataset

# ----- LOAD DATA -----
class ECGDataset(Dataset):
    def __init__(self, data_dir):
        self.records = [f[:-4] for f in os.listdir(data_dir) if f.endswith('.dat')]
        self.data_dir = data_dir

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        signals, fields = wfdb.rdsamp(os.path.join(self.data_dir, record))
        signals = resample(signals[:, 0], SEQ_LEN)  # Resample to fixed length
        signals = (signals - np.mean(signals)) / np.std(signals)  # Normalize
        label = 1 if "abnormal" in record else 0  # Binary anomaly label
        return torch.tensor(signals, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Load dataset
dataset = ECGDataset(DATA_DIR)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ----- MODEL -----
class TransformerModel(nn.Module):
    def __init__(self, input_dim=1, d_model=128, nhead=4, num_layers=2, num_classes=2):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)  # Add input dimension
        x = self.embed(x)  # Project to d_model
        x = self.transformer(x)  # Pass through Transformer layers
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)

# Initialize model
model = TransformerModel().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ----- TRAIN -----
def train():
    model.train()
    for epoch in range(EPOCHS):
        total_loss, correct, total = 0, 0, 0
        for signals, labels in train_loader:
            signals, labels = signals.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        print(f"Epoch {epoch+1}: Loss {total_loss:.4f}, Accuracy {correct/total:.4f}")

# ----- EVALUATE -----
def evaluate():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for signals, labels in test_loader:
            signals, labels = signals.to(DEVICE), labels.to(DEVICE)
            outputs = model(signals)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    print(f"Test Accuracy: {correct/total:.4f}")

# ----- RUN -----
if __name__ == "__main__":
    train()
    evaluate()
