"""Main script for running multimodal experiments on combined dataset."""

import torch
import os
from config import MultimodalConfig
from data_preprocessing import load_data, compute_class_weights
from fully_connected_network import FullyConnectedNetwork
from experiment_runner import run_experiments_over_seeds
from results_analysis import analyze_experiment_results
from multimodal_utils import set_seed, print_device_info

def main():
    """Main function to run the complete multimodal experiment."""
    
    # Print device information
    print_device_info()
    
    # Load configuration
    config = MultimodalConfig()
    
    # Set initial seed for data loading reproducibility
    set_seed(42)
    
    print("Loading data...")
    # Load images, labels, texts, and dataframe
    images, labels, texts, df = load_data(config)
    
    print("Computing class weights...")
    # Compute class weights for handling imbalance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights_tensor = compute_class_weights(labels, device)
    
    print("Setting up fully connected network...")
    # Create fully connected network for final classification
    fc_network = FullyConnectedNetwork(
        input_dim=config.fc_input_dim,
        hidden_dim=config.fc_hidden_dim,
        output_dim=config.output_dim
    )
    
    print("Starting experiments across multiple seeds...")
    print(f"Seeds to be used: {config.seeds}")
    
    # Run experiments across multiple seeds
    val_results, test_results, test_auc_roc = run_experiments_over_seeds(
        seed_list=config.seeds,
        df=df,
        images=images,
        cnn_weight_path=config.densenet_weight_path,
        vit_weight_path=config.deit_weight_path,
        bert_weight_path=config.bert_weight_path,
        fc_network=fc_network,
        class_weights_tensor=class_weights_tensor,
        config=config
    )
    
    print("\nAnalyzing results...")
    # Analyze results with bootstrap confidence intervals
    summary_df, individual_df = analyze_experiment_results(
        test_reports=test_results,
        auc_roc_scores=test_auc_roc,
        seed_list=config.seeds,
        save_results=True,
        output_dir='./results/'
    )
    
    print("\nExperiments completed successfully!")
    print(f"Summary statistics saved to ./results/summary_results.csv")
    print(f"Individual results saved to ./results/individual_results.csv")
    
    return summary_df, individual_df

if __name__ == "__main__":
    # Ensure results directory exists
    os.makedirs('./results/', exist_ok=True)
    
    try:
        summary_results, individual_results = main()
        print("\nAll experiments completed successfully!")
    except Exception as e:
        print(f"An error occurred during experiments: {str(e)}")
        raise