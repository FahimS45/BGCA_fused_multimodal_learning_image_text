import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random


##================= INbreast text data =================##

# Load the dataset
csv_file = "/inbreast/Text-data-annotation.csv" 
df = pd.read_csv(csv_file)

# Prepare the dataset
label_mapping = {'Non-Malignant': 0, 'Malignant': 1}
labels = df['Class'].map(label_mapping).values
texts = df['Structured Text'].tolist()

##================= Discriminative word detection and removal =================##

def filter_discriminative_words_all_classes(texts, labels, top_n=20, custom_exclusions={'no'}):
    custom_stopwords = ENGLISH_STOP_WORDS.difference(custom_exclusions)

    # Tokenize and clean text by class
    class_tokens = {0: [], 1: []}
    for text, label in zip(texts, labels):
        words = text.lower().split()
        words = [word.strip('.,():;-"\n') for word in words if word not in custom_stopwords]
        class_tokens[label].extend(words)

    # Count word frequencies
    freq_0 = Counter(class_tokens[0])
    freq_1 = Counter(class_tokens[1])

    removed_words_by_class = {0: set(), 1: set()}

    for target_class in [0, 1]:
        target_freq = freq_0 if target_class == 0 else freq_1
        other_freq = freq_1 if target_class == 0 else freq_0

        discriminative_scores = {}
        for word, freq in target_freq.items():
            if freq > 1:
                score = freq / (1 + other_freq.get(word, 0))
                discriminative_scores[word] = score

        top_words = sorted(discriminative_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_discriminative_words = {word for word, _ in top_words}
        removed_words_by_class[target_class] = top_discriminative_words

    # Combine all removed words
    all_removed_words = removed_words_by_class[0].union(removed_words_by_class[1])

    # Remove all discriminative words from texts
    texts_filtered = []
    for text in texts:
        words = text.split()
        cleaned_words = [
            word for word in words
            if word.lower().strip('.,():;-"\n') not in all_removed_words
        ]
        cleaned_text = ' '.join(cleaned_words)
        texts_filtered.append(cleaned_text)

    return texts_filtered, removed_words_by_class

##================= Dataset class =================##

class InbreastTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

##================= Train, evaluation, and test loops =================##    

def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    losses = []
    predictions = []
    true_labels = []

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        # Backward pass
        losses.append(loss.item())
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer and scheduler step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Predictions and true labels
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds)
        true_labels.extend(labels)

    predictions = torch.stack(predictions).cpu()
    true_labels = torch.stack(true_labels).cpu()

    # Return average loss and classification report
    return np.mean(losses), classification_report(
        true_labels, predictions, target_names=df['Class'].unique(), output_dict=True
    )


def eval_model(model, data_loader, device):
    model.eval()
    losses = []
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            losses.append(loss.item())

            # Predictions and true labels
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds)
            true_labels.extend(labels)

    predictions = torch.stack(predictions).cpu()
    true_labels = torch.stack(true_labels).cpu()

    # Return average loss and classification report
    return np.mean(losses), classification_report(
        true_labels, predictions, target_names=df['Class'].unique(), output_dict=True
    )


def test_model(model, data_loader, device, seed=None):
    model.eval()
    losses = []
    predictions = []
    true_labels = []
    probabilities = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

            losses.append(loss.item())

            preds = torch.argmax(logits, dim=1)
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

    # Confusion Matrix
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

    # Results DataFrame
    results_df = pd.DataFrame([{
        'Seed': seed if seed is not None else 'N/A',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc_roc
    }])

    return results_df, class_report

##================= Experiments across 10 different seeds =================##

def run_experiments_over_seeds_bert(seed_list, texts, labels):
    
    test_results = []
    test_auc_roc = []

    batch_size = 4
    num_epochs = 50
    patience_limit = 5

    for seed in seed_list:
        print(f"\n🔁 Running experiment with seed: {seed}")
        set_seed(seed)

        def seed_worker(worker_id):
            np.random.seed(seed)
            random.seed(seed)

        # Split data into train+val and test sets
        texts_train_val, texts_test, labels_train_val, labels_test = train_test_split(
            texts, labels, test_size=0.15, stratify=labels, random_state=seed
        )
        
        # Split train+val into train and val sets
        texts_train, texts_val, labels_train, labels_val = train_test_split(
            texts_train_val, labels_train_val, test_size=0.176, stratify=labels_train_val, random_state=seed
        )
        
        # Top words removal appearing in one class only
        texts_train_filtered, removed_words_by_class = filter_discriminative_words_all_classes(
            texts_train, labels_train, top_n=7
        )

        # You can inspect the removed words:
        print("\nSummary of Removed Words by Class:")
        for cls, words in removed_words_by_class.items():
            print(f"Class {cls}: {sorted(words)}")

        
        # Initialize the tokenizer and model
        tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        model = BertForSequenceClassification.from_pretrained(
            'emilyalsentzer/Bio_ClinicalBERT', 
            num_labels=df['Class'].nunique()
        )

        # Dataset
        train_dataset = InbreastTextDataset(texts_train_filtered, labels_train, tokenizer)
        val_dataset = InbreastTextDataset(texts_val, labels_val, tokenizer)
        test_dataset = InbreastTextDataset(texts_test, labels_test, tokenizer)

        # Dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=total_steps)

        # === Early Stopping Training Loop ===
        best_val_f1 = 0
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            train_loss, train_report = train_epoch(model, train_loader, optimizer, device, scheduler)
            val_loss, val_report = eval_model(model, val_loader, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_report['accuracy']:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_report['accuracy']:.4f}")

            val_f1 = val_report['macro avg']['f1-score']
            print(f"Val Macro F1: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                print("F1 improved. Saving model...")
                best_val_f1 = val_f1
                torch.save(model.state_dict(), "best_bert_model_state.bin")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience_limit}")
                if patience_counter >= patience_limit:
                    print("Early stopping triggered.")
                    break

        # Load best model and evaluate
        model.load_state_dict(torch.load("best_bert_model_state.bin"))
        test_results_df, test_report = test_model(model, test_loader, device)

        print(f"Classification Report:\n{test_report}")


        test_results.append(test_report)
        test_auc_roc.append(test_results_df['AUC-ROC'])

        # Plot Loss Curve
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Seed: {seed} Loss Curve')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Final resultse)
    return test_results, test_auc_roc

##================= Results summary =================##

def summarize_classification_results(test_reports, test_auc_roc, n_bootstrap=10000, ci=95):
    
    macro_f1_scores = []
    accuracies = []
    macro_precisions = []
    macro_recalls = []
    
    for report in test_reports:
        macro_f1_scores.append(report["macro avg"]["f1-score"])
        macro_precisions.append(report["macro avg"]["precision"])
        macro_recalls.append(report["macro avg"]["recall"])
        accuracies.append(report["accuracy"])
    
    auc_roc_scores = [score.values[0] for score in test_auc_roc]

    print("===== Individual Seed Run Results =====")
    for i, (acc, f1, prec, rec, roc_auc) in enumerate(zip(accuracies, macro_f1_scores, macro_precisions, macro_recalls, auc_roc_scores), 1):
        print(f"Seed Run {i}:")
        print(f"  Accuracy        : {acc:.4f}")
        print(f"  Macro F1-score  : {f1:.4f}")
        print(f"  Macro Precision : {prec:.4f}")
        print(f"  Macro Recall    : {rec:.4f}")
        print(f"  AUC-ROC         : {roc_auc:.4f}")
        print()

    metrics_df = pd.DataFrame({
        'Accuracy': accuracies,
        'Precision': macro_precisions,
        'Recall': macro_recalls,
        'F1-Score': macro_f1_scores,
        'AUC-ROC': auc_roc_scores
    })

    def bootstrap_ci(data, n_bootstrap=n_bootstrap, ci=ci):
        data = np.array(data)
        if len(data) == 0:
            return np.nan, np.nan, np.nan
        means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)]
        lower = np.percentile(means, (100 - ci) / 2)
        upper = np.percentile(means, 100 - (100 - ci) / 2)
        return np.mean(means), lower, upper

    summary_rows = []
    for metric in metrics_df.columns:
        mean_val = metrics_df[metric].mean()
        std_val = metrics_df[metric].std()
        boot_mean, ci_lower, ci_upper = bootstrap_ci(metrics_df[metric].dropna().values)
        summary_rows.append({
            'Metric': metric,
            'Mean': mean_val,
            'Std Dev': std_val,
            'Boot Mean': boot_mean,
            f'{ci}% CI Lower': ci_lower,
            f'{ci}% CI Upper': ci_upper
        })

    summary_df = pd.DataFrame(summary_rows)

    print("---")
    print(f"===== Aggregated Results (Mean ± Std & Bootstrap {ci}% CI) =====")
    for _, row in summary_df.iterrows():
        print(f"{row['Metric']}:")
        print(f"  Mean ± Std       : {row['Mean']:.4f} ± {row['Std Dev']:.4f}")
        print(f"  {ci}% CI (Bootstrap): [{row[f'{ci}% CI Lower']:.4f}, {row[f'{ci}% CI Upper']:.4f}]")
        print()

    return summary_df    

##================= The main function =================##

def main():

    seed_list = [42, 77, 7, 101, 314, 2024, 123, 88, 11, 999]

    test_reports, test_auc_roc = run_experiments_over_seeds_bert(
        seed_list=seed_list,
        texts=texts,
        labels=labels,
    )

    summary_df = summarize_classification_results(
        test_reports,
        test_auc_roc,
        n_bootstrap=10000,
        ci=95
    )

    print("Final Aggregated Summary:")
    print(summary_df)


if __name__ == "__main__":
    main()



