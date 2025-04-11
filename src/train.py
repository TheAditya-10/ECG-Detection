import torch
import torch.optim as optim
import torch.nn as nn
from preprocessing import get_dataloaders
from anomaly_detection import get_model
from classification import get_classifier
from tqdm import tqdm  # Import tqdm for progress bar

BATCH_SIZE = 100
EPOCHS = 5
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = get_dataloaders(BATCH_SIZE)

# Load models
anomaly_model = get_model(DEVICE)
classifier_model = get_classifier(DEVICE)

# Loss functions
criterion_anomaly = nn.CrossEntropyLoss()
criterion_classification = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(
    list(anomaly_model.parameters()) + list(classifier_model.parameters()), lr=LR
)

def validate():
    """Validation loop to evaluate models on the test set."""
    anomaly_model.eval()
    classifier_model.eval()
    total_loss, correct_anomaly, correct_class, total = 0, 0, 0, 0

    with torch.no_grad():  # Disable gradient computation for validation
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
    train()
