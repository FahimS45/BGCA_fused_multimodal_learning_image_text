"""Main experiment script for Inbreast vision classification."""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import DeiTForImageClassification
from vision_config import Config
from vision_utils import set_seed
from vision_dataset import (load_and_preprocess_data, split_data, get_transforms, 
                            create_dataloaders)
from vision_trainer import train_model, evaluate_model
from vision_results import summarize_classification_results

def run_single_experiment(config: Config, seed: int, X, y):
    """Run experiment for single seed."""
    print(f"\n🔁 Running experiment with seed: {seed}")
    set_seed(seed)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, config.test_size, config.val_size, seed
    )
    
    # Create dataloaders (ADASYN is applied inside)
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test,
        train_transform, val_test_transform, config.batch_size, seed
    )
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeiTForImageClassification.from_pretrained(config.model_name)
    model.classifier = nn.Linear(model.classifier.in_features, config.num_classes)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 
                          lr=config.learning_rate, 
                          weight_decay=config.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                         step_size=config.step_size, 
                                         gamma=config.gamma)
    
    # Train model
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                config.num_epochs, config.patience, device)
    
    # Load best model and evaluate
    model.load_state_dict(torch.load("deit_model.pth"))
    test_metrics, test_report = evaluate_model(model, test_loader, criterion, device, seed)
    
    return test_metrics, test_report

def main():
    """Run complete multi-seed experiment."""
    config = Config()
    
    # Load data once
    print("Loading data...")
    X, y = load_and_preprocess_data(config.data_dir, config.classes, config.target_size)
    
    test_metrics_list = []
    test_reports_list = []
    test_auc_roc = []
    
    for seed in config.seeds:
        try:
            metrics, report = run_single_experiment(config, seed, X, y)
            test_metrics_list.append(metrics)
            test_reports_list.append(report)
            test_auc_roc.append(metrics['auc_roc'])
        except Exception as e:
            print(f"Seed {seed} failed: {e}")
            continue
    
    # Summarize results
    summary_df = summarize_classification_results(
        test_reports_list, 
        test_auc_roc,
        config.n_bootstrap,
        config.ci
    )
    
    print("Final Aggregated Summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()