"""Results analysis and summarization for vision model."""

import numpy as np
import pandas as pd
from typing import List, Dict

def summarize_results(test_reports: List[Dict], test_auc_roc: List[float], 
                                   n_bootstrap: int = 10000, ci: int = 95) -> pd.DataFrame:
    """Summarize classification results across multiple seed runs."""
    
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