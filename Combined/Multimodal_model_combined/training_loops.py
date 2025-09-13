"""Training, validation, and testing loops for multimodal model."""

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, Optional

def train_model(model: torch.nn.Module, 
                train_loader: torch.utils.data.DataLoader, 
                criterion: torch.nn.Module, 
                optimizer: torch.optim.Optimizer, 
                scheduler: torch.optim.lr_scheduler._LRScheduler, 
                device: torch.device) -> Tuple[float, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: The multimodal model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer for updating model parameters
        scheduler: Learning rate scheduler
        device: Device to run training on
        
    Returns:
        Tuple of (epoch_loss, epoch_accuracy)
    """
    model.train()  
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch in tqdm(train_loader, desc="Training"):
        # Move data to the device
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, input_ids, attention_mask)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        running_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_samples

    # Step the scheduler every epoch
    scheduler.step()

    return epoch_loss, epoch_accuracy

def evaluate_model(model: torch.nn.Module, 
                   val_loader: torch.utils.data.DataLoader, 
                   criterion: torch.nn.Module, 
                   device: torch.device,
                   class_names: list = None) -> Tuple[float, float, Dict]:
    """
    Evaluate the model on validation data.
    
    Args:
        model: The multimodal model to evaluate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run evaluation on
        class_names: Names of the classes for classification report
        
    Returns:
        Tuple of (epoch_loss, epoch_accuracy, classification_report)
    """
    model.eval()  
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    predictions = []
    true_labels = []

    with torch.no_grad(): 
        for batch in tqdm(val_loader, desc="Validation"):
            # Move data to the device
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)

            # Accumulate metrics
            running_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            predictions.extend(preds.cpu())
            true_labels.extend(labels.cpu())

    predictions = torch.stack(predictions)
    true_labels = torch.stack(true_labels)
    
    epoch_loss = running_loss / len(val_loader)
    epoch_accuracy = correct_predictions / total_samples

    if class_names is None:
        class_names = ['Non-Malignant', 'Malignant']

    return epoch_loss, epoch_accuracy, classification_report(
        true_labels, predictions, target_names=class_names, output_dict=True
    )

def test_model(model: torch.nn.Module, 
               test_loader: torch.utils.data.DataLoader, 
               criterion: torch.nn.Module, 
               device: torch.device, 
               seed: Optional[int] = None, 
               save_cm_path: str = '/tmp/confusion_matrix.pdf',
               class_names: list = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Test the model and generate comprehensive results.
    
    Args:
        model: The multimodal model to test
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to run testing on
        seed: Random seed used (for reporting)
        save_cm_path: Path to save confusion matrix
        class_names: Names of the classes
        
    Returns:
        Tuple of (results_dataframe, classification_report_dict)
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    predictions = []
    true_labels = []
    probabilities = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images, input_ids, attention_mask)
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
    auc_roc = roc_auc_score(true_labels, probabilities[:, 1], multi_class="ovr") if probabilities.shape[1] > 1 else None

    print(f"Test Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if auc_roc is not None:
        print(f"AUC-ROC: {auc_roc:.4f}")

    # Generate confusion matrix
    generate_confusion_matrix(true_labels, predictions, save_cm_path, class_names)

    if class_names is None:
        class_names = ['Non-Malignant', 'Malignant']

    # Classification Report
    class_report = classification_report(
        true_labels,
        predictions,
        target_names=class_names,
        output_dict=True
    )

    # Return metrics summary and full classification report
    results_df = pd.DataFrame([{
        'Seed': seed if seed is not None else 'N/A',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc_roc
    }])

    return results_df, class_report

def generate_confusion_matrix(true_labels: torch.Tensor, 
                            predictions: torch.Tensor, 
                            save_path: str, 
                            class_names: list = None) -> Dict:
    """
    Generate and save confusion matrix plot.
    
    Args:
        true_labels: True class labels
        predictions: Predicted class labels
        save_path: Path to save the confusion matrix plot
        class_names: Names of the classes
        
    Returns:
        Classification report dictionary
    """
    if class_names is None:
        class_names = ['Non-Malignant', 'Malignant']

    # Generate Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
                yticklabels=class_names, linewidths=2, cbar=False, square=True, annot_kws={"size": 14})
    
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)

    # Save confusion matrix
    plt.savefig(save_path, bbox_inches="tight", format="pdf")
    plt.show()
    plt.close()
    print(f"Confusion matrix saved as {save_path}")

    # Return classification report
    return classification_report(true_labels, predictions, target_names=class_names, output_dict=True)