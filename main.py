import os
from src.config.config import set_device
from src.data.data_processing import load_and_preprocess_data
from src.models.training import train_and_evaluate
from src.utils.visualization import plot_all_visualizations
from src.evaluation.metrics import compute_epsilon

def main():
    # Configuration
    tasks = ["sst2", "cola", "mrpc"]
    train_sizes = {"sst2": 3000, "cola": 5000, "mrpc": 2000}
    eval_sizes = {"sst2": 500, "cola": 500, "mrpc": 500}
    gpu_id = 2
    epochs = 10
    lr = 5e-5
    batch_size = 16
    noise_multiplier = 0.1
    strategies = ["SGD", "DP-SGD", "DND", "Adaptive Clipping"]
    
    # Set device
    device = set_device(gpu_id)
    
    # Initialize results dictionary
    results = {}
    
    for task in tasks:
        print(f"\nProcessing task: {task}")
        
        # Load and preprocess data
        train_loader, eval_loader, tokenizer = load_and_preprocess_data(
            task, train_sizes[task], eval_sizes[task], batch_size
        )
        
        # Train and evaluate for each strategy
        task_results, losses_dict, grad_norms_dict, uncertainty_dict = train_and_evaluate(
            task, train_loader, eval_loader, strategies, device, tokenizer,
            epochs, lr, noise_multiplier
        )
        
        # Store results
        results[task] = task_results
        
        # Generate visualizations
        plot_all_visualizations(
            task, strategies, losses_dict, grad_norms_dict, uncertainty_dict,
            task_results, tasks, noise_multiplier
        )
    
    # Print final results
    for task in tasks:
        print(f"\nFinal Results for {task}:")
        for strategy, result in results[task].items():
            print(f"{strategy}: Train Accuracy: {result['train_accuracy']:.4f}, "
                  f"Eval Accuracy: {result['eval_accuracy']:.4f}, "
                  f"F1: {result['eval_f1']:.4f}, "
                  f"Epsilon: {result['epsilon'] if result['epsilon'] else 'N/A'}")

if __name__ == "__main__":
    main()