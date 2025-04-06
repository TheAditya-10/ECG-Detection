from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score
import torch
import torch.nn as nn
from tqdm import tqdm
from preprocessing import get_dataloaders
from anomaly_detection import get_model
from classification import get_classifier

# Hyperparameters and device setup
BATCH_SIZE = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the test data
_, test_loader = get_dataloaders(BATCH_SIZE)

# Load the saved models
anomaly_model = get_model(DEVICE)
classifier_model = get_classifier(DEVICE)
anomaly_model.load_state_dict(torch.load("models/anomaly_model.pth"))
classifier_model.load_state_dict(torch.load("models/classifier_model.pth"))

# Set models to evaluation mode
anomaly_model.eval()
classifier_model.eval()

# Define loss functions
criterion_anomaly = nn.CrossEntropyLoss()
criterion_classification = nn.CrossEntropyLoss()

def test():
    total_loss = 0
    all_anomaly_labels = []
    all_anomaly_preds = []
    all_class_labels = []
    all_class_preds = []

    # Wrap the DataLoader with tqdm for a progress bar
    progress_bar = tqdm(test_loader, desc="Testing", unit="batch")

    with torch.no_grad():  # Disable gradient computation for testing
        for signals, anomaly_labels, class_labels in progress_bar:
            signals, anomaly_labels, class_labels = (
                signals.to(DEVICE), anomaly_labels.to(DEVICE), class_labels.to(DEVICE)
            )

            # Forward pass
            anomaly_preds = anomaly_model(signals)
            class_preds = classifier_model(signals)

            # Calculate losses
            loss_anomaly = criterion_anomaly(anomaly_preds, anomaly_labels)
            loss_classification = criterion_classification(class_preds, class_labels)
            loss = loss_anomaly + loss_classification

            total_loss += loss.item()

            # Store predictions and labels for metrics
            all_anomaly_labels.extend(anomaly_labels.cpu().numpy())
            all_anomaly_preds.extend(anomaly_preds.argmax(dim=1).cpu().numpy())
            all_class_labels.extend(class_labels.cpu().numpy())
            all_class_preds.extend(class_preds.argmax(dim=1).cpu().numpy())

            # Update progress bar with current loss
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    # Calculate metrics for anomaly detection
    print("\nAnomaly Detection Metrics:")
    print(f"Accuracy: {accuracy_score(all_anomaly_labels, all_anomaly_preds):.4f}")
    print(f"Recall: {recall_score(all_anomaly_labels, all_anomaly_preds, average='weighted'):.4f}")
    print(f"F1-Score: {f1_score(all_anomaly_labels, all_anomaly_preds, average='weighted'):.4f}")

    # Calculate metrics for classification
    print("\nClassification Metrics:")
    print(f"Accuracy: {accuracy_score(all_class_labels, all_class_preds):.4f}")
    print(f"Recall: {recall_score(all_class_labels, all_class_preds, average='weighted'):.4f}")
    print(f"F1-Score: {f1_score(all_class_labels, all_class_preds, average='weighted'):.4f}")

    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(all_class_labels, all_class_preds))

if __name__ == "__main__":
    test()