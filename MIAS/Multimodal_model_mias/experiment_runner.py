"""Experiment runner for multiple seed experiments."""

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from typing import List, Tuple, Dict

from config import MultimodalConfig
from model_loaders import load_densenet_model, load_deit_model, load_bert_model
from multimodal_model import MultiModalModel
from fully_connected_network import FullyConnectedNetwork
from multimodal_dataset import MultiModalDataset
from training_utils import set_seed, train_model, evaluate_model, test_model, bootstrap_ci

def run_experiments_over_seeds(config: MultimodalConfig, texts: List[str], 
                              images: List, labels: List[int],
                              train_transform, val_test_transform) -> Tuple[List[Dict], List[float]]:
    """
    Run experiments across multiple seeds.
    
    Args:
        config: Configuration object
        texts: List of text descriptions
        images: List of preprocessed images
        labels: List of labels
        train_transform: Training image transformations
        val_test_transform: Validation/test image transformations
        
    Returns:
        Tuple of (test_reports, test_auc_roc_scores)
    """
    test_reports = []
    test_auc_roc_scores = []
    
    # Compute class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=torch.unique(torch.tensor(labels)).numpy(),
        y=labels
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights_tensor = class_weights_tensor.to(device)
    
    for seed in config.seeds:
        print(f"\n====== Running for Seed: {seed} ======")
        
        # Set seed
        set_seed(seed)
        def seed_worker(worker_id): 
            import numpy as np
            import random
            np.random.seed(seed)
            random.seed(seed)

        # Load models fresh for this seed
        densenet_model = load_densenet_model(config.densenet_weight_path)
        deit_model = load_deit_model(config.deit_weight_path, config.deit_model_name)
        text_model, bert_tokenizer = load_bert_model(config.bert_weight_path, config.bert_model_name)

        # Data split
        texts_train_val, texts_test, images_train_val, images_test, labels_train_val, labels_test = train_test_split(
            texts, images, labels, test_size=config.test_size, stratify=labels, random_state=seed
        )
        texts_train, texts_val, images_train, images_val, labels_train, labels_val = train_test_split(
            texts_train_val, images_train_val, labels_train_val, 
            test_size=config.val_size, stratify=labels_train_val, random_state=seed
        )

        # Create datasets and dataloaders
        train_dataset = MultiModalDataset(images_train, texts_train, labels_train, 
                                        bert_tokenizer, train_transform, config.max_length)
        val_dataset = MultiModalDataset(images_val, texts_val, labels_val, 
                                      bert_tokenizer, val_test_transform, config.max_length)
        test_dataset = MultiModalDataset(images_test, texts_test, labels_test, 
                                       bert_tokenizer, val_test_transform, config.max_length)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, worker_init_fn=seed_worker)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, worker_init_fn=seed_worker)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, worker_init_fn=seed_worker)

        # Create fully connected network
        fc_network = FullyConnectedNetwork(
            input_dim=config.fc_input_dim,
            hidden_dim=config.fc_hidden_dim,
            output_dim=config.output_dim
        )

        # Initialize model
        model = MultiModalModel(densenet_model, deit_model, text_model, fc_network).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=config.label_smoothing)

        # Optimizer: only update trainable parameters (attention + fc)
        optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
        scheduler = StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

        # Training with early stopping
        best_val_f1 = 0
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(config.num_epochs):
            train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, scheduler, device)
            val_loss, val_acc, val_report = evaluate_model(model, val_loader, criterion, device)
            val_f1 = val_report['macro avg']['f1-score']

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1:03d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), f"best_model_seed_{seed}.bin")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print("Early stopping.")
                    break

        # Load best model and test
        model.load_state_dict(torch.load(f"best_model_seed_{seed}.bin"))
        results_dict, test_report = test_model(model, test_loader, criterion, device, seed)

        test_reports.append(test_report)
        test_auc_roc_scores.append(results_dict['AUC-ROC'])

        # Plot loss curves
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss (Seed {seed})')
        plt.legend()
        plt.grid(True)
        plt.tight