"""Training and evaluation utilities."""

import torch
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report)
from typing import Tuple, Dict

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_model(model, train_loader, criterion, optimizer, scheduler, device) -> Tuple[float, float]:
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch in tqdm(train_loader, desc="Training"):
        # Move data to device
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, input_ids, attention_mask)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        running_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_samples

    # Step scheduler
    scheduler.step()

    return epoch_loss, epoch_accuracy

def evaluate_model(model, val_loader, criterion, device) -> Tuple[float, float, Dict]:
    """Evaluate the model."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Move data to device
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

            predictions.extend(preds)
            true_labels.extend(labels)

    predictions = torch.stack(predictions).cpu()
    true_labels = torch.stack(true_labels).cpu()

    epoch_loss = running_loss / len(val_loader)
    epoch_accuracy = correct_predictions / total_samples

    class_report = classification_report(
        true_labels, predictions, 
        target_names=['Benign', 'Malignant'], 
        output_dict=True
    )

    return epoch_loss, epoch_accuracy, class_report

def test_model(model, test_loader, criterion, device, seed=None) -> Tuple[Dict, Dict]:
    """Test the model and return comprehensive metrics."""
    model.eval()
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
            _, preds = torch.max(outputs, dim=1)

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
    auc_roc = roc_auc_score(true_labels, probabilities[:, 1]) if probabilities.shape[1] > 1 else None

    # Results dictionary
    results = {
        'Seed': seed if seed is not None else 'N/A',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc_roc
    }

    # Classification report
    class_report = classification_report(
        true_labels, predictions,
        target_names=['Benign', 'Malignant'],
        output_dict=True
    )

    return results, class_report

def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 10000, ci: int = 95) -> Tuple[float, float, float]:
    """Calculate bootstrap confidence interval."""
    data = np.array(data)
    means = []
    n = len(data)
    
    if n == 0:
        return np.nan, np.nan, np.nan
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(sample))
    
    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    
    return np.mean(means), lower, upper