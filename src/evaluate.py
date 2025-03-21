import torch
from preprocessing import get_dataloaders
from anomaly_detection import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, test_loader = get_dataloaders()
model = get_model(DEVICE)
model.load_state_dict(torch.load("models/anomaly_model.pth"))
model.eval()

def evaluate():
    correct, total = 0, 0
    with torch.no_grad():
        for signals, labels in test_loader:
            signals, labels = signals.to(DEVICE), labels.to(DEVICE)
            outputs = model(signals)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    print(f"Test Accuracy: {correct/total:.4f}")

if __name__ == "__main__":
    evaluate()
