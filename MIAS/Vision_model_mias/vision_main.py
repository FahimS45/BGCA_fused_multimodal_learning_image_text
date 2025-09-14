"""Main experiment script for MIAS classification."""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

from mammography_config import Config
from mammography_utils import set_seed, calculate_class_weights
from mammography_dataset import (
    load_and_preprocess_data, split_data, get_transforms, 
    create_dataloaders, augment_minority_classes
)
from mammography_trainer import full_training_loop, test_model, plot_training_curves
from mammography_results import summarize_classification_results


def run_single_experiment(config: Config, seed: int, X, y):
    """Run experiment for single seed."""
    print(f"\n🔁 Running experiment with seed: {seed}")
    set_seed(seed)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms()
    
    # Apply data augmentation if enabled
    if config.augment_minority:
        print("Applying minority class augmentation...")
        X, y = augment_minority_classes(X, y, train_transform)
        
        # Show class distribution after augmentation
        class_counts = Counter(y)
        print("Class distribution after augmentation:", class_counts)
        
        # Plot class distribution
        counts = [class_counts[i] for i in sorted(class_counts.keys())]
        class_names = ['Benign', 'Malignant']
        plt.figure(figsize=(8, 5))
        plt.bar(class_names, counts, color=['blue', 'red'])
        plt.title(f"Class Distribution After Augmentation (Seed {seed})")
        plt.xlabel("Classes")
        plt.ylabel("Number of Samples")
        plt.show()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=config.test_size, val_size=config.val_size, seed=seed
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test,
        train_transform, val_test_transform, config.batch_size, seed
    )
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if config.model_name == "densenet121":
        model = models.densenet121(pretrained=True)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, config.num_classes)
    else:
        raise ValueError(f"Model {config.model_name} not supported")
    
    model = model.to(device)
    
    # Calculate class weights
    class_weights = calculate_class_weights(y_train, device)
    print(f"Class weights: {class_weights}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.label_smoothing)
    optimizer = optim.Adam(model.parameters(), 
                          lr=config.learning_rate, 
                          weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                        step_size=config.step_size, 
                                        gamma=config.gamma)
    
    # Train model
    train_losses, val_losses = full_training_loop(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, config.num_epochs, config.patience
    )
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, seed)
    
    # Load best model and evaluate
    model.load_state_dict(torch.load("best_model.pth"))
    test_metrics, test_report = test_model(model, test_loader, criterion, device, seed)
    
    return test_metrics, test_report


def run_experiments_over_seeds(config: Config, X, y):
    """Run experiments over multiple seeds."""
    val_results = []
    test_results = []
    test_auc_roc = []

    for seed in config.seeds:
        try:
            metrics_df, class_report = run_single_experiment(config, seed, X, y)
            val_results.append(metrics_df)
            test_results.append(class_report)
            
            # Extract AUC-ROC score
            auc_roc = metrics_df["AUC-ROC"].values[0] if not metrics_df.empty else None
            test_auc_roc.append(auc_roc)
            
        except Exception as e:
            print(f"Seed {seed} failed: {e}")
            continue
    
    return val_results, test_results, test_auc_roc


def main():
    """Run complete multi-seed experiment."""
    config = Config()
    
    # Load data once
    print("Loading and preprocessing mammography data...")
    X, y = load_and_preprocess_data(config)
    
    # Display initial class distribution
    class_counts = Counter(y)
    print("Initial class distribution:", class_counts)
    
    # Plot initial class distribution
    counts = [class_counts[i] for i in sorted(class_counts.keys())]
    class_names = ['Benign', 'Malignant']
    plt.figure(figsize=(8, 5))
    plt.bar(class_names, counts, color=['lightblue', 'lightcoral'])
    plt.title("Initial Class Distribution")
    plt.xlabel("Classes")
    plt.ylabel("Number of Samples")
    plt.show()
    
    print(f"Loaded {len(X)} images with {len(set(y))} classes")
    
    # Run experiments
    print(f"\n🚀 Starting experiments with {len(config.seeds)} seeds...")
    val_results, test_reports, test_auc_roc = run_experiments_over_seeds(config, X, y)
    
    # Summarize results
    if test_reports:
        print(f"\n📊 Completed {len(test_reports)} successful experiments")
        summary_df = summarize_classification_results(
            test_reports, 
            test_auc_roc,
            config.n_bootstrap,
            config.ci
        )
        
        print("\n" + "="*60)
        print("FINAL AGGREGATED SUMMARY")
        print("="*60)
        print(summary_df.to_string(index=False, float_format='%.4f'))
    else:
        print("❌ No successful experiments completed.")


if __name__ == "__main__":
    main()