import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import ADASYN
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torchvision import transforms
from transformers import DeiTForImageClassification, DeiTFeatureExtractor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import seaborn as sns
import random


# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed profiling
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

##================= Image dataset =================##

def load_and_preprocess_data(data_dir, target_size=(224, 224)):
    classes = ['Non-Malignant_', 'Malignant_']
    X = []
    y = []

    # Loop through the classes (Non-Malignant and Malignant)
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        
        # Loop through the images in the class directory
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            
            # Open and convert image to RGB
            img = Image.open(img_path).convert('RGB')
            
            # Resize the image to target_size
            img = img.resize(target_size)
            
            # Append the resized PIL Image object to X
            X.append(img)
            
            # Append the corresponding class label to y
            y.append(i)

    return X, np.array(y)

data_dir = '/inbreast/Inbreast-images-png'
X, y = load_and_preprocess_data(data_dir)

##================= ADASYN =================##

def apply_adasyn(X, y):
    # Convert PIL Images to numpy arrays
    X_arrays = [np.array(img) for img in X]
    
    # Get the shape of the images
    img_shape = X_arrays[0].shape
    
    # Reshape the arrays to 2D for ADASYN
    X_reshaped = np.array([x.flatten() for x in X_arrays])
    
    # Apply ADASYN
    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X_reshaped, y)
    
    # Reshape back to original image shape
    X_balanced = [x.reshape(img_shape) for x in X_resampled]
    
    # Convert back to PIL Images
    X_balanced = [Image.fromarray(x.astype('uint8')) for x in X_balanced]
    
    return X_balanced, y_resampled

##================= Dataset class =================##

class InbreastDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

##================= Augmentation =================##  

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(60),
    #transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

##================= Train and evaluation loops =================##  

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=5):
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        correct = 0
        total = 0
        
        # Training loop
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Forward pass and loss calculation
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Average train loss for the epoch
        avg_train_loss = running_train_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(avg_train_loss)
        
        # Validation loop
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).logits
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Average validation loss for the epoch
        avg_val_loss = running_val_loss / len(val_loader)
        val_acc = correct / total
        val_losses.append(avg_val_loss)
        
        print(f"--> Epoch Number : {epoch + 1} | Training Loss : {round(avg_train_loss, 4)} | Validation Loss : {round(avg_val_loss, 4)} | Train Accuracy : {round(train_acc * 100, 2)}% | Validation Accuracy : {round(val_acc * 100, 2)}%")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print("Saving model....")
            torch.save(model.state_dict(), "deit_model.pth")  # Save the best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    # Plotting training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.show()


def evaluate_model(model, test_loader, criterion, device, seed=None):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    predictions = []
    true_labels = []
    probabilities = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            # Handle both tensor and dict output
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            loss = criterion(logits, labels)
            running_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, dim=1)

            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

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

    # Confusion matrix
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

##================= Experiments across 10 different seeds =================##

def run_experiments_over_seeds_deit(seed_list, X, y, train_transform, val_test_transform):
    val_results = []
    test_results = []
    test_auc_roc = []

    batch_size = 32

    for seed in seed_list:
        print(f"\n🔁 Running experiment with seed: {seed}")
        set_seed(seed)

        def seed_worker(worker_id):
            np.random.seed(seed)
            random.seed(seed)

        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, seed)

        # Apply adasyn
        X_balanced, y_balanced = apply_adasyn(X_train, y_train)

        train_dataset = InbreastDataset(X_balanced, y_balanced, transform=train_transform)
        val_dataset = InbreastDataset(X_val, y_val, transform=val_test_transform)
        test_dataset = InbreastDataset(X_test, y_test, transform=val_test_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker)

        # Load DeiT model pretrained
        model = DeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224")
        model.classifier = nn.Linear(model.classifier.in_features, 2)  # Adjust classifier
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

        train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=5)

        model.load_state_dict(torch.load("deit_model.pth"))

        result_df, class_report = evaluate_model(model, test_loader, criterion, device, seed)

        val_results.append(result_df)
        test_results.append(class_report)
        test_auc_roc.append(result_df["AUC-ROC"].values[0])

    val_results_df = pd.concat(val_results, ignore_index=True)

    return val_results_df, test_results, test_auc_roc

##================= Results summary =================##

def summarize_classification_results(test_reports, test_auc_roc, n_bootstrap=10000, ci=95):
  
    # Extract metrics 
    macro_f1_scores = []
    accuracies = []
    macro_precisions = []
    macro_recalls = []

    for report in test_reports:
        macro_f1_scores.append(report["macro avg"]["f1-score"])
        macro_precisions.append(report["macro avg"]["precision"])
        macro_recalls.append(report["macro avg"]["recall"])
        accuracies.append(report["accuracy"])

    auc_roc_scores = test_auc_roc

    print("===== Individual Seed Run Results =====")
    for i, (acc, f1, prec, rec, roc_auc) in enumerate(
        zip(accuracies, macro_f1_scores, macro_precisions, macro_recalls, auc_roc_scores), 1
    ):
        print(f"Seed Run {i}:")
        print(f"  Accuracy        : {acc:.4f}")
        print(f"  Macro F1-score  : {f1:.4f}")
        print(f"  Macro Precision : {prec:.4f}")
        print(f"  Macro Recall    : {rec:.4f}")
        print(f"  AUC-ROC         : {roc_auc:.4f}")
        print()

    # Convert to DataFrame 
    metrics_df = pd.DataFrame({
        'Accuracy': accuracies,
        'Precision': macro_precisions,
        'Recall': macro_recalls,
        'F1-Score': macro_f1_scores,
        'AUC-ROC': auc_roc_scores
    })

    # Bootstrap CI function
    def bootstrap_ci(data, n_bootstrap=n_bootstrap, ci=ci):
        data = np.array(data)
        n = len(data)
        if n == 0:
            return np.nan, np.nan, np.nan
        means = [np.mean(np.random.choice(data, size=n, replace=True)) for _ in range(n_bootstrap)]
        lower = np.percentile(means, (100 - ci) / 2)
        upper = np.percentile(means, 100 - (100 - ci) / 2)
        return np.mean(means), lower, upper

    # Aggregate results 
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
    val_results, test_reports, test_auc_roc = run_experiments_over_seeds_deit(
        seed_list=seed_list,
        X=X,
        y=y,
        train_transform=train_transform,
        val_test_transform=val_test_transform
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

