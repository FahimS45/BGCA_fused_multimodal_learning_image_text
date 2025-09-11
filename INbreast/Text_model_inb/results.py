"""Results analysis and summarization."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any

def summarize_results(test_reports: List[Dict], test_metrics: List[Dict], 
                     n_bootstrap: int = 10000, ci: int = 95) -> pd.DataFrame:
    """Summarize results across multiple seed runs."""
    
    # Extract metrics
    accuracies = [m['accuracy'] for m in test_metrics]
    precisions = [m['precision'] for m in test_metrics]
    recalls = [m['recall'] for m in test_metrics]
    f1_scores = [m['f1'] for m in test_metrics]
    auc_scores = [m['auc_roc'] for m in test_metrics if m['auc_roc'] is not None]
    
    print("===== Individual Results =====")
    for i, (acc, prec, rec, f1, auc) in enumerate(zip(accuracies, precisions, recalls, f1_scores, auc_scores), 1):
        print(f"Run {i}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    
    # Calculate summary statistics
    metrics_data = {
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1-Score': f1_scores,
        'AUC-ROC': auc_scores
    }
    
    summary_rows = []
    for metric, values in metrics_data.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Bootstrap CI
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        ci_lower = np.percentile(bootstrap_means, (100 - ci) / 2)
        ci_upper = np.percentile(bootstrap_means, 100 - (100 - ci) / 2)
        
        summary_rows.append({
            'Metric': metric,
            'Mean': mean_val,
            'Std': std_val,
            f'{ci}% CI Lower': ci_lower,
            f'{ci}% CI Upper': ci_upper
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    print(f"\n===== Summary Results (Bootstrap {ci}% CI) =====")
    for _, row in summary_df.iterrows():
        print(f"{row['Metric']}: {row['Mean']:.4f} ± {row['Std']:.4f} "
              f"[{row[f'{ci}% CI Lower']:.4f}, {row[f'{ci}% CI Upper']:.4f}]")
    
    return summary_df