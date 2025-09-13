"""Main experiment script for DMID text classification."""

import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from config import Config
from utils import set_seed
from dataset import load_data, split_data, create_dataloaders
from trainer import train_epoch, eval_epoch, test_model
from results import summarize_results

def run_single_experiment(config: Config, seed: int):
    """Run experiment for single seed."""
    print(f"\n🔁 Running experiment with seed: {seed}")
    set_seed(seed)
    
    # Load and split data
    texts, labels = load_data(config.csv_file, config.text_col, config.label_col, config.label_mapping)
    texts_train, texts_val, texts_test, labels_train, labels_val, labels_test = split_data(
        texts, labels, config.test_size, config.val_size, seed
    )
    
    # Initialize model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    model = BertForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        texts_train, texts_val, texts_test,
        labels_train, labels_val, labels_test,
        tokenizer, config.batch_size, config.max_length
    )
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=total_steps)
    
    # Training loop with early stopping
    best_val_f1 = 0
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        train_loss, train_report = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_report = eval_epoch(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_report['accuracy']:.4f} | Val Acc: {val_report['accuracy']:.4f}")
        
        val_f1 = val_report['macro avg']['f1-score']
        print(f"Val F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            print("F1 improved. Saving model...")
            best_val_f1 = val_f1
            torch.save(model.state_dict(), f"best_model_seed_{seed}.bin")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print("Early stopping triggered.")
                break
    
    # Load best model and test
    model.load_state_dict(torch.load(f"best_model_seed_{seed}.bin"))
    test_metrics, test_report = test_model(model, test_loader, device)
    
    return test_metrics, test_report

def main():
    """Run complete multi-seed experiment."""
    config = Config()
    
    test_metrics_list = []
    test_reports_list = []
    
    for seed in config.seeds:
        try:
            metrics, report = run_single_experiment(config, seed)
            test_metrics_list.append(metrics)
            test_reports_list.append(report)
        except Exception as e:
            print(f"Seed {seed} failed: {e}")
            continue
    
    # Summarize results
    summary_df = summarize_results(test_reports_list, test_metrics_list)
    
    print("\nFinal Summary:")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()