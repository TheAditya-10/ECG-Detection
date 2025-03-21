import torch
import torch.optim as optim
import torch.nn as nn
from preprocessing import get_dataloaders
from anomaly_detection import get_model

# ----- CONFIG -----
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_loader, test_loader = get_dataloaders(BATCH_SIZE)

# Initialize model
model = get_model(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

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
    torch.save(model.state_dict(), "models/anomaly_model.pth")

if __name__ == "__main__":
    train()
