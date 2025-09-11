"""Training and evaluation functions for vision model."""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, classification_report)
import seaborn as sns
import pandas as pd
from typing import Tuple, Dict

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs: int = 100, patience: int = 5, device=None):
    """Train the model with early stopping."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        correct = 0
        total = 0
        
        # Training loop
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Forward pass and loss calculation
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Average train loss for the epoch
        avg_train_loss = running_train_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(avg_train_loss)
        
        # Validation loop
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).logits
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Average validation loss for the epoch
        avg_val_loss = running_val_loss / len(val_loader)
        val_acc = correct / total
        val_losses.append(avg_val_loss)
        
        print(f"--> Epoch Number : {epoch + 1} | Training Loss : {round(avg_train_loss, 4)} | "
              f"Validation Loss : {round(avg_val_loss, 4)} | Train Accuracy : {round(train_acc * 100, 2)}% | "
              f"Validation Accuracy : {round(val_acc * 100, 2)}%")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print("Saving model....")
            torch.save(model.state_dict(), "deit_model.pth")  # Save the best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    # Plotting training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.show()

def evaluate_model(model, test_loader, criterion, device, seed=None):
    """Evaluate the model and return comprehensive metrics."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    predictions = []
    true_labels = []
    probabilities = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            # Handle both tensor and dict output
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            loss = criterion(logits, labels)
            running_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, dim=1)

            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

            predictions.extend(preds.cpu())
            true_labels.extend(labels.cpu())
            probabilities.extend(probs.cpu())

    predictions = torch.stack(predictions)
    true_labels = torch.stack(true_labels)
    probabilities = torch.stack(probabilities)

    # Metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    f1 = f1_score(true_labels, predictions, average='macro')
    auc_roc = roc_auc_score(true_labels, probabilities[:, 1], multi_class="ovr") if probabilities.shape[1] > 1 else None

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if auc_roc is not None:
        print(f"AUC-ROC: {auc_roc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-malignant', 'Malignant'],
                yticklabels=['Non-malignant', 'Malignant'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification Report
    class_report = classification_report(
        true_labels,
        predictions,
        target_names=['Non-malignant', 'Malignant'],
        output_dict=True
    )

    # Return metrics summary and full classification report
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc
    }
    
    results_df = pd.DataFrame([{
        'Seed': seed if seed is not None else 'N/A',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc_roc
    }])

    return metrics, class_report