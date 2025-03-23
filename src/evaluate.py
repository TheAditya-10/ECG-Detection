import torch
import torch.nn as nn
from preprocessing import get_dataloaders
from anomaly_detection import get_model
from classification import get_classifier
from sklearn.metrics import accuracy_score, classification_report

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
anomaly_model = get_model(DEVICE)
classifier_model = get_classifier(DEVICE)

anomaly_model.load_state_dict(torch.load("models/anomaly_model.pth"))
classifier_model.load_state_dict(torch.load("models/classifier_model.pth"))

anomaly_model.eval()
classifier_model.eval()

# Load test data
_, test_loader = get_dataloaders(batch_size=32)

def evaluate():
    all_anomaly_preds, all_anomaly_labels = [], []
    all_class_preds, all_class_labels = []

    with torch.no_grad():
        for signals, anomaly_labels, class_labels in test_loader:
            signals, anomaly_labels, class_labels = (
                signals.to(DEVICE), anomaly_labels.to(DEVICE), class_labels.to(DEVICE)
            )
            
            # Predictions
            anomaly_preds = anomaly_model(signals).argmax(dim=1)
            class_preds = classifier_model(signals).argmax(dim=1)

            # Store results
            all_anomaly_preds.extend(anomaly_preds.cpu().numpy())
            all_anomaly_labels.extend(anomaly_labels.cpu().numpy())
            all_class_preds.extend(class_preds.cpu().numpy())
            all_class_labels.extend(class_labels.cpu().numpy())

    # Compute Metrics
    anomaly_acc = accuracy_score(all_anomaly_labels, all_anomaly_preds)
    class_acc = accuracy_score(all_class_labels, all_class_preds)

    print(f"Anomaly Detection Accuracy: {anomaly_acc:.4f}")
    print(f"Classification Accuracy: {class_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_class_labels, all_class_preds, digits=4))

if __name__ == "__main__":
    evaluate()
