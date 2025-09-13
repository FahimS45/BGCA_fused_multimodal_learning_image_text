"""Training and evaluation functions."""

import torch
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import numpy as np
from typing import Tuple, Dict

def train_epoch(model, loader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    losses = []
    predictions, true_labels = [], []

    for batch in tqdm(loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu())
        true_labels.extend(labels.cpu())

    predictions = torch.stack(predictions).numpy()
    true_labels = torch.stack(true_labels).numpy()
    
    return np.mean(losses), classification_report(true_labels, predictions, output_dict=True)

def eval_epoch(model, loader, device):
    """Evaluate for one epoch."""
    model.eval()
    losses = []
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            losses.append(loss.item())
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu())
            true_labels.extend(labels.cpu())

    predictions = torch.stack(predictions).numpy()
    true_labels = torch.stack(true_labels).numpy()
    
    return np.mean(losses), classification_report(true_labels, predictions, output_dict=True)

def test_model(model, loader, device):
    """Test model and return comprehensive metrics."""
    model.eval()
    predictions, true_labels, probabilities = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu())
            true_labels.extend(labels.cpu())
            probabilities.extend(probs.cpu())

    predictions = torch.stack(predictions).numpy()
    true_labels = torch.stack(true_labels).numpy()
    probabilities = torch.stack(probabilities).numpy()

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    f1 = f1_score(true_labels, predictions, average='macro')
    auc_roc = roc_auc_score(true_labels, probabilities[:, 1]) if probabilities.shape[1] == 2 else None

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc
    }
    
    report = classification_report(true_labels, predictions, output_dict=True)
    
    return metrics, report