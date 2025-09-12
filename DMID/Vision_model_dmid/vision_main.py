"""Main experiment script for vision classification."""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from vision_config import Config
from vision_utils import set_seed, calculate_class_weights
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
        X, y, test_size=config.test_size, val_size=config.val_size, seed=seed
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test,
        train_transform, val_test_transform, config.batch_size, seed
    )
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if config.model_name == "resnet152":
        model = models.resnet152(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, config.num_classes)
    else:
        raise ValueError(f"Model {config.model_name} not supported")
    
    model = model.to(device)
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_loader, device)
    print(f"Class weights: {class_weights}")
    
    # Loss and optimizer with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), 
                          lr=config.learning_rate, 
                          weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                        step_size=config.step_size, 
                                        gamma=config.gamma)
    
    # Train model
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                config.num_epochs, config.patience)
    
    # Load best model and evaluate
    model.load_state_dict(torch.load("best_model.pth"))
    test_metrics, test_report = evaluate_model(model, test_loader, criterion, device, seed)
    
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
            test_auc_roc.append(metrics_df["AUC-ROC"].values[0])
        except Exception as e:
            print(f"Seed {seed} failed: {e}")
            continue
    
    val_results_df = pd.concat(val_results, ignore_index=True) if val_results else None
    return val_results_df, test_results, test_auc_roc

def main():
    """Run complete multi-seed experiment."""
    config = Config()
    
    # Load data once
    print("Loading data...")
    X, y = load_and_preprocess_data(config.data_dir, config.classes, config.target_size)
    print(f"Loaded {len(X)} images with {len(set(y))} classes")
    
    # Run experiments
    val_results_df, test_reports, test_auc_roc = run_experiments_over_seeds(config, X, y)
    
    # Summarize results
    if test_reports:
        summary_df = summarize_classification_results(
            test_reports, 
            test_auc_roc,
            config.n_bootstrap,
            config.ci
        )
        
        print("\nFinal Aggregated Summary:")
        print(summary_df.to_string(index=False))
    else:
        print("No successful experiments completed.")

if __name__ == "__main__":
    main()