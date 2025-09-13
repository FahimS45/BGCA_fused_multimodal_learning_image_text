"""Statistical analysis and bootstrap confidence intervals for experimental results."""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

def bootstrap_ci(data: List[float], n_bootstrap: int = 10000, ci: int = 95) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence intervals for a dataset.
    
    Args:
        data: List of values to compute CI for
        n_bootstrap: Number of bootstrap samples
        ci: Confidence interval percentage
        
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    data = np.array(data)
    means = []
    n = len(data)
    if n == 0:
        return np.nan, np.nan, np.nan
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(sample))
    
    lower = np.percentile(means, (100 - ci) / 2)
    upper_percentile = 100 - (100 - ci) / 2
    upper = np.percentile(means, upper_percentile)
    
    return np.mean(means), lower, upper

def evaluate_metrics_with_bootstrap(test_reports: List[Dict], 
                                   auc_roc_scores: List[float], 
                                   n_bootstrap: int = 10000, 
                                   ci: int = 95) -> pd.DataFrame:
    """
    Evaluate metrics across multiple seed runs with bootstrap confidence intervals.
    
    Args:
        test_reports: List of classification report dictionaries
        auc_roc_scores: List of AUC-ROC scores
        n_bootstrap: Number of bootstrap samples
        ci: Confidence interval percentage
        
    Returns:
        DataFrame containing summary statistics
    """
    macro_f1_scores = []
    accuracies = []
    macro_precisions = []
    macro_recalls = []

    for report in test_reports:
        macro_f1_scores.append(report["macro avg"]["f1-score"])
        macro_precisions.append(report["macro avg"]["precision"])
        macro_recalls.append(report["macro avg"]["recall"])
        accuracies.append(report["accuracy"])

    # Create metrics DataFrame
    metrics_df = pd.DataFrame({
        'Accuracy': accuracies,
        'Precision': macro_precisions,
        'Recall': macro_recalls,
        'F1-Score': macro_f1_scores,
        'AUC-ROC': auc_roc_scores
    })

    metrics_to_bootstrap = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']

    summary_rows = []
    for metric in metrics_to_bootstrap:
        values = metrics_df[metric].dropna().values
        mean_val = np.mean(values)
        std_val = np.std(values)
        boot_mean, ci_lower, ci_upper = bootstrap_ci(values, n_bootstrap=n_bootstrap, ci=ci)
        summary_rows.append({
            'Metric': metric,
            'Mean': mean_val,
            'Std Dev': std_val,
            'Boot Mean': boot_mean,
            f'{ci}% CI Lower': ci_lower,
            f'{ci}% CI Upper': ci_upper
        })

    summary_df = pd.DataFrame(summary_rows)

    # Print individual results
    print("===== Individual Seed Run Results =====")
    for i, (acc, f1, prec, rec, roc_auc) in enumerate(zip(
        metrics_df['Accuracy'], metrics_df['F1-Score'], metrics_df['Precision'],
        metrics_df['Recall'], metrics_df['AUC-ROC']), 1):
        print(f"Seed Run {i}:")
        print(f"  Accuracy        : {acc:.4f}")
        print(f"  Macro F1-score  : {f1:.4f}")
        print(f"  Macro Precision : {prec:.4f}")
        print(f"  Macro Recall    : {rec:.4f}")
        print(f"  AUC-ROC         : {roc_auc:.4f}\n")

    # Print the aggregated summary
    print("===== Aggregated Results (Mean ± Std & Bootstrap CI) =====")
    for _, row in summary_df.iterrows():
        print(f"{row['Metric']}:")
        print(f"  Mean ± Std        : {row['Mean']:.4f} ± {row['Std Dev']:.4f}")
        print(f"  {ci}% CI (Bootstrap): [{row[f'{ci}% CI Lower']:.4f}, {row[f'{ci}% CI Upper']:.4f}]\n")

    return summary_df

def save_results_to_csv(summary_df: pd.DataFrame, 
                       individual_results: pd.DataFrame, 
                       output_dir: str = './results/'):
    """
    Save experimental results to CSV files.
    
    Args:
        summary_df: Summary statistics DataFrame
        individual_results: Individual seed results DataFrame
        output_dir: Directory to save results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    summary_df.to_csv(f"{output_dir}/summary_results.csv", index=False)
    individual_results.to_csv(f"{output_dir}/individual_results.csv", index=False)
    
    print(f"Results saved to {output_dir}")

def create_individual_results_df(test_reports: List[Dict], 
                                auc_roc_scores: List[float], 
                                seed_list: List[int]) -> pd.DataFrame:
    """
    Create DataFrame with individual seed results.
    
    Args:
        test_reports: List of classification report dictionaries
        auc_roc_scores: List of AUC-ROC scores
        seed_list: List of seeds used
        
    Returns:
        DataFrame containing individual results
    """
    individual_results = []
    
    for i, (report, auc_score, seed) in enumerate(zip(test_reports, auc_roc_scores, seed_list)):
        individual_results.append({
            'Run': i + 1,
            'Seed': seed,
            'Accuracy': report["accuracy"],
            'Precision': report["macro avg"]["precision"],
            'Recall': report["macro avg"]["recall"],
            'F1-Score': report["macro avg"]["f1-score"],
            'AUC-ROC': auc_score
        })
    
    return pd.DataFrame(individual_results)

def analyze_experiment_results(test_reports: List[Dict], 
                             auc_roc_scores: List[float], 
                             seed_list: List[int], 
                             save_results: bool = True,
                             output_dir: str = './results/') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete analysis of experimental results.
    
    Args:
        test_reports: List of classification report dictionaries
        auc_roc_scores: List of AUC-ROC scores
        seed_list: List of seeds used
        save_results: Whether to save results to files
        output_dir: Directory to save results
        
    Returns:
        Tuple of (summary_df, individual_results_df)
    """
    # Create individual results DataFrame
    individual_df = create_individual_results_df(test_reports, auc_roc_scores, seed_list)
    
    # Compute summary statistics with bootstrap CI
    summary_df = evaluate_metrics_with_bootstrap(test_reports, auc_roc_scores)
    
    # Save results if requested
    if save_results:
        save_results_to_csv(summary_df, individual_df, output_dir)
    
    return summary_df, individual_df