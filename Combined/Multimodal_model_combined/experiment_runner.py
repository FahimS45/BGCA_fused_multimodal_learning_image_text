"""Experiment runner for multiple seed experiments."""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

from config import MultimodalConfig
from multimodal_utils import set_seed
from model_loaders import load_densenet_model, load_deit_model, load_bert_model
from multimodal_model import MultiModalModelGatedCrossAttention
from fully_connected_network import FullyConnectedNetwork
from multimodal_dataset import MultiModalDataset
from data_preprocessing import preprocess_multimodal_data
from training_loops import train_model, evaluate_model, test_model

def run_experiments_over_seeds(seed_list: List[int], 
                             df, 
                             images: List, 
                             cnn_weight_path: str, 
                             vit_weight_path: str, 
                             bert_weight_path: str, 
                             fc_network: FullyConnectedNetwork, 
                             class_weights_tensor: torch.Tensor,
                             config: MultimodalConfig) -> Tuple[List, List, List]:
    """
    Run experiments across multiple seeds for statistical significance.
    
    Args:
        seed_list: List of random seeds to use
        df: DataFrame containing text data and metadata
        images: List of loaded images
        cnn_weight_path: Path to CNN (DenseNet) weights
        vit_weight_path: Path to ViT (DeiT) weights
        bert_weight_path: Path to BERT weights
        fc_network: Fully connected network for classification
        class_weights_tensor: Class weights for handling imbalance
        config: Configuration object
        
    Returns:
        Tuple of (validation_reports, test_reports, test_auc_roc_scores)
    """
    all_val_reports = []
    all_test_reports = []
    all_test_auc_roc_scores = []
    
    for seed in seed_list:
        print(f"\n====== Running for Seed: {seed} ======")
        
        # Set seed for reproducibility
        def seed_worker(worker_id): 
            np.random.seed(seed)
            random.seed(seed)
        set_seed(seed)
        
        # Reload all models fresh for this seed
        densenet_model = load_densenet_model(cnn_weight_path)
        deit_model = load_deit_model(vit_weight_path, config.deit_model_name)
        text_model, bert_tokenizer = load_bert_model(bert_weight_path, config.bert_model_name)
        
        # Data splitting with proper handling of INbreast groups
        (texts_train, labels_train, images_train,
        texts_val, labels_val, images_val,
        texts_test, labels_test, images_test) = preprocess_multimodal_data(df, images, seed=seed)

        # Create transforms
        from data_preprocessing import create_transforms
        train_transform, val_test_transform = create_transforms()

        # Create datasets
        train_dataset = MultiModalDataset(images_train, texts_train, labels_train, bert_tokenizer, train_transform, config.max_length)
        val_dataset = MultiModalDataset(images_val, texts_val, labels_val, bert_tokenizer, val_test_transform, config.max_length)
        test_dataset = MultiModalDataset(images_test, texts_test, labels_test, bert_tokenizer, val_test_transform, config.max_length)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, worker_init_fn=seed_worker)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, worker_init_fn=seed_worker)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, worker_init_fn=seed_worker)

        # Model and training setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MultiModalModelGatedCrossAttention(
            resnet_model=densenet_model,  # Note: using densenet instead of resnet
            deit_model=deit_model, 
            text_model=text_model, 
            fc_network=fc_network,
            resnet_dim=config.densenet_dim,
            deit_dim=config.deit_dim,
            text_dim=config.text_dim,
            hidden_dim=config.hidden_dim
        ).to(device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=config.label_smoothing)
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                        lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

        # Training with early stopping
        best_val_f1 = 0
        patience_counter = 0
        best_model_state = None

        # Store loss history
        train_losses = []
        val_losses = []

        for epoch in range(config.num_epochs):
            train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, scheduler, device)
            val_loss, val_acc, val_report = evaluate_model(model, val_loader, criterion, device)
            val_f1 = val_report['macro avg']['f1-score']

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1:03d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

            # Early stopping logic
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print("Early stopping triggered.")
                    break

        # Load best model and evaluate on test set
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Test evaluation
        results_df_test, test_report = test_model(
            model, test_loader, criterion, device, seed=seed,
            save_cm_path=f'/tmp/confusion_matrix_seed_{seed}.pdf'
        )

        all_val_reports.append(val_report)
        all_test_reports.append(test_report)
        
        # Extract AUC-ROC score
        if not results_df_test.empty and 'AUC-ROC' in results_df_test.columns:
            auc_roc_score = results_df_test['AUC-ROC'].iloc[0]
            all_test_auc_roc_scores.append(auc_roc_score)
        else:
            print(f"Warning: AUC-ROC score not found in results for seed {seed}")
            all_test_auc_roc_scores.append(None)

        # Plot training curves
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss (Seed {seed})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return all_val_reports, all_test_reports, all_test_auc_roc_scores