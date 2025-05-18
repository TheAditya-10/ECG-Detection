import torch
import torch.optim as optim
import torch.nn as nn
from preprocessing import get_dataloaders
from anomaly_detection import get_model
from classification import get_classifier
from tqdm import tqdm
import numpy as np

BATCH_SIZE = 100
EPOCHS = 5
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = get_dataloaders(BATCH_SIZE)

def compute_weighted_loss(train_loader, device, num_anomaly_classes=2, num_class_classes=10):
    """Compute class weights for anomaly and classification tasks with progress bar."""
    all_anomaly_labels = []
    all_class_labels = []
    dataset = train_loader.dataset
    desc = "Collecting labels for class weights"
    for _, anomaly_label, class_label in tqdm(dataset, desc=desc, unit="sample"):
        all_anomaly_labels.append(anomaly_label.item())
        all_class_labels.append(class_label.item())

    # Print present and missing classes for debugging
    present_classes = np.unique(all_class_labels)
    all_possible_classes = np.arange(num_class_classes)
    missing_classes = set(all_possible_classes) - set(present_classes)
    print("Present classes:", present_classes)
    print("Missing classes:", missing_classes)

    # Anomaly weights (always 2 classes)
    anomaly_counts = np.bincount(all_anomaly_labels, minlength=num_anomaly_classes)
    anomaly_weights = np.zeros(num_anomaly_classes, dtype=np.float32)
    nonzero = anomaly_counts > 0
    anomaly_weights[nonzero] = 1.0 / anomaly_counts[nonzero]
    anomaly_weights = anomaly_weights / anomaly_weights.sum()  # Normalize

    # Class weights (always 10 classes)
    class_counts = np.bincount(all_class_labels, minlength=num_class_classes)
    class_weights = np.zeros(num_class_classes, dtype=np.float32)
    nonzero = class_counts > 0
    class_weights[nonzero] = 1.0 / class_counts[nonzero]
    class_weights = class_weights / class_weights.sum()  # Normalize

    # Convert to torch tensors and move to device
    anomaly_weights = torch.tensor(anomaly_weights, dtype=torch.float32).to(device)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    return anomaly_weights, class_weights

# Compute weighted loss using the function (with progress bar)
anomaly_weights, class_weights = compute_weighted_loss(train_loader, DEVICE, num_anomaly_classes=2, num_class_classes=10)

# Load models
anomaly_model = get_model(DEVICE)
classifier_model = get_classifier(DEVICE)

# Loss functions (apply weights)
criterion_anomaly = nn.CrossEntropyLoss(weight=anomaly_weights)
criterion_classification = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer for updating weights
optimizer = optim.Adam(
    list(anomaly_model.parameters()) + list(classifier_model.parameters()), lr=LR
)

def validate():
    """Validation loop to evaluate models on the test set."""
    anomaly_model.eval()
    classifier_model.eval()
    total_loss, correct_anomaly, correct_class, total = 0, 0, 0, 0

    with torch.no_grad():
        for signals, anomaly_labels, class_labels in test_loader:
            signals, anomaly_labels, class_labels = (
                signals.to(DEVICE), anomaly_labels.to(DEVICE), class_labels.to(DEVICE)
            )

            anomaly_preds = anomaly_model(signals)
            class_preds = classifier_model(signals)

            loss_anomaly = criterion_anomaly(anomaly_preds, anomaly_labels)
            loss_classification = criterion_classification(class_preds, class_labels)
            loss = loss_anomaly + loss_classification

            total_loss += loss.item()
            correct_anomaly += (anomaly_preds.argmax(dim=1) == anomaly_labels).sum().item()
            correct_class += (class_preds.argmax(dim=1) == class_labels).sum().item()
            total += anomaly_labels.size(0)

    anomaly_acc = correct_anomaly / total
    class_acc = correct_class / total
    avg_loss = total_loss / len(test_loader)

    print(f"Validation: Loss {avg_loss:.4f}, Anomaly Acc {anomaly_acc:.4f}, Class Acc {class_acc:.4f}")
    return avg_loss, anomaly_acc, class_acc

def train():
    best_anomaly_acc = 0.0
    best_class_acc = 0.0

    for epoch in range(EPOCHS):
        anomaly_model.train()
        classifier_model.train()
        total_loss, correct_anomaly, correct_class, total = 0, 0, 0, 0

        # Wrap the DataLoader with tqdm for a progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")

        for signals, anomaly_labels, class_labels in progress_bar:
            signals, anomaly_labels, class_labels = (
                signals.to(DEVICE), anomaly_labels.to(DEVICE), class_labels.to(DEVICE)
            )

            optimizer.zero_grad()  # Zero the parameter gradients
            anomaly_preds = anomaly_model(signals)
            class_preds = classifier_model(signals)

            loss_anomaly = criterion_anomaly(anomaly_preds, anomaly_labels)
            loss_classification = criterion_classification(class_preds, class_labels)
            loss = loss_anomaly + loss_classification

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct_anomaly += (anomaly_preds.argmax(dim=1) == anomaly_labels).sum().item()
            correct_class += (class_preds.argmax(dim=1) == class_labels).sum().item()
            total += anomaly_labels.size(0)

            # Update progress bar with current loss
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                anomaly_acc=f"{correct_anomaly/total:.4f}",
                class_acc=f"{correct_class/total:.4f}"
            )

        print(f"Epoch {epoch+1}: Loss {total_loss:.4f}, Anomaly Acc {correct_anomaly/total:.4f}, Class Acc {correct_class/total:.4f}")

        # Validate the models
        val_loss, val_anomaly_acc, val_class_acc = validate()

        # Save the best models
        if val_anomaly_acc > best_anomaly_acc:
            best_anomaly_acc = val_anomaly_acc
            torch.save(anomaly_model.state_dict(), "models/best_anomaly_model.pth")
            print(f"Saved best anomaly model with accuracy {best_anomaly_acc:.4f}")

        if val_class_acc > best_class_acc:
            best_class_acc = val_class_acc
            torch.save(classifier_model.state_dict(), "models/best_classifier_model.pth")
            print(f"Saved best classifier model with accuracy {best_class_acc:.4f}")

if __name__ == "__main__":
    print("Training and testing the model...")
    train()
