"""Training and evaluation functions for MIAS classification."""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, classification_report)
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report


def train_model(model, train_loader, criterion, optimizer, scheduler, device):
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch in tqdm(train_loader, desc="Training"):
        images = batch[0].to(device)
        labels = batch[1].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_samples
    scheduler.step()

    return epoch_loss, epoch_accuracy


def evaluate_model(model, val_loader, criterion, device, class_names):
    """Evaluate model on validation set."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            images = batch[0].to(device)
            labels = batch[1].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

            predictions.extend(preds.cpu())
            true_labels.extend(labels.cpu())

    epoch_loss = running_loss / len(val_loader)
    epoch_accuracy = correct_predictions / total_samples

    report = classification_report(true_labels, predictions, target_names=class_names, output_dict=True)
    return epoch_loss, epoch_accuracy, report


def test_model(model, data_loader, criterion, device, seed=None):
    """Test model and return comprehensive metrics."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    predictions = []
    true_labels = []
    probabilities = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Testing"):
            images = batch[0].to(device)
            labels = batch[1].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

            predictions.extend(preds.cpu())
            true_labels.extend(labels.cpu())

            probs = torch.softmax(outputs, dim=1)
            probabilities.extend(probs.cpu())

    predictions = torch.stack(predictions)
    true_labels = torch.stack(true_labels)
    probabilities = torch.stack(probabilities)

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    f1 = f1_score(true_labels, predictions, average='macro')

    # AUC-ROC for binary classification
    auc_roc = None
    if probabilities.shape[1] > 1:
        try:
            auc_roc = roc_auc_score(true_labels, probabilities[:, 1], multi_class="ovr")
        except ValueError:
            auc_roc = None

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if auc_roc is not None:
        print(f"AUC-ROC: {auc_roc:.4f}")

    # Plot confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-malignant', 'Malignant'],
                yticklabels=['Non-malignant', 'Malignant'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification report
    class_report = classification_report(
        true_labels,
        predictions,
        target_names=['Non-malignant', 'Malignant'],
        output_dict=True
    )

    # Results DataFrame
    results_df = pd.DataFrame([{
        'Seed': seed if seed is not None else 'N/A',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc_roc
    }])

    return results_df, class_report


def full_training_loop(model, train_loader, val_loader, criterion, optimizer, 
                      scheduler, device, num_epochs: int = 100, patience: int = 10):
    """Complete training loop with early stopping."""
    best_val_f1 = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    class_names = ['Non-malignant', 'Malignant']

    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_acc, val_report = evaluate_model(model, val_loader, criterion, device, class_names)
        val_f1 = val_report['macro avg']['f1-score']

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1:03d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return train_losses, val_losses


def plot_training_curves(train_losses, val_losses, seed):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss (Seed {seed})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()