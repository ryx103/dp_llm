import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

def plot_training_curves(strategies, losses_dict, grad_norms_dict, task):
    """
    Plot training loss and gradient norm curves.
    
    Args:
        strategies (list): List of training strategies.
        losses_dict (dict): Dictionary of losses per strategy.
        grad_norms_dict (dict): Dictionary of gradient norms per strategy.
        task (str): GLUE task name.
    """
    plt.figure(figsize=(12, 5))
    for strategy in strategies:
        plt.subplot(1, 2, 1)
        plt.plot(losses_dict[strategy], label=f"{strategy} Loss")
        plt.subplot(1, 2, 2)
        plt.plot(grad_norms_dict[strategy], label=f"{strategy} Grad Norm")
    
    plt.subplot(1, 2, 1)
    plt.title(f"Training Loss - {task}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.title(f"Gradient Norm - {task}")
    plt.xlabel("Epoch")
    plt.ylabel("Grad Norm")
    plt.legend()
    
    plt.savefig(f"/u/yra7pn/llm_dp/training_curves_{task}.png")
    plt.close()

def plot_confusion_matrix(cm, strategy, task):
    """
    Plot confusion matrix for a strategy and task.
    
    Args:
        cm: Confusion matrix.
        strategy (str): Training strategy.
        task (str): GLUE task name.
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {strategy} on {task}")
    plt.savefig(f"/u/yra7pn/llm_dp/cm_{strategy}_{task}.png")
    plt.close()

def plot_performance_comparison(results, tasks, metric_name="eval_accuracy"):
    """
    Plot performance comparison across tasks and strategies.
    
    Args:
        results (dict): Results dictionary.
        tasks (list): List of GLUE tasks.
        metric_name (str): Metric to plot (eval_accuracy or eval_f1).
    """
    plt.figure(figsize=(10, 6))
    bar_width = 0.2
    for i, task in enumerate(tasks):
        for j, strategy in enumerate(results[task].keys()):
            plt.bar(i + j * bar_width, results[task][strategy][metric_name], 
                    bar_width, label=f"{strategy}" if i == 0 else None)
    
    plt.xlabel("Tasks")
    plt.ylabel(metric_name.replace('_', ' ').capitalize())
    plt.title(f"{metric_name.replace('_', ' ').capitalize()} Comparison Across Tasks and Strategies")
    plt.xticks(np.arange(len(tasks)) + bar_width * 1.5, tasks)
    plt.legend()
    plt.savefig(f"/u/yra7pn/llm_dp/performance_comparison_{metric_name}.png")
    plt.close()

def plot_privacy_performance_tradeoff(results, tasks):
    """
    Plot privacy-performance tradeoff scatter plot.
    
    Args:
        results (dict): Results dictionary.
        tasks (list): List of GLUE tasks.
    """
    plt.figure(figsize=(10, 6))
    for task in tasks:
        epsilons = [results[task][s]["epsilon"] for s in results[task] if results[task][s]["epsilon"] is not None]
        accuracies = [results[task][s]["eval_accuracy"] for s in results[task] if results[task][s]["epsilon"] is not None]
        plt.scatter(epsilons, accuracies, label=task)
    
    plt.xlabel("Epsilon (Privacy Budget)")
    plt.ylabel("Eval Accuracy")
    plt.title("Privacy-Performance Tradeoff")
    plt.legend()
    plt.savefig(f"/u/yra7pn/llm_dp/privacy_performance_tradeoff_{tasks}.png")
    plt.close()

def plot_gradient_noise_sensitivity(grad_norms_dict, noise_multiplier=0.1, clip_norm=2.0, task="sst2"):
    """
    Plot gradient noise sensitivity by epoch.
    
    Args:
        grad_norms_dict (dict): Dictionary of gradient norms per strategy.
        noise_multiplier (float): Noise multiplier for DP strategies.
        clip_norm (float): Gradient clipping norm.
        task (str): GLUE task name.
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(list(grad_norms_dict.values())[0]) + 1)
    
    for strategy in grad_norms_dict.keys():
        grad_norms = grad_norms_dict[strategy]
        plt.plot(epochs, grad_norms, label=f'{strategy} Grad Norm', linewidth=2)
        
        if strategy == 'DP-SGD':
            noise_scale = noise_multiplier * clip_norm
            lower_bound = [max(0, norm - noise_scale) for norm in grad_norms]
            upper_bound = [norm + noise_scale for norm in grad_norms]
            plt.fill_between(epochs, lower_bound, upper_bound, alpha=0.2, label='DP-SGD Noise Range')
        elif strategy == 'DND':
            noise_scales = [noise_multiplier * (0.95 ** epoch) * clip_norm for epoch in range(len(grad_norms))]
            lower_bound = [max(0, grad_norms[e] - noise_scales[e]) for e in range(len(grad_norms))]
            upper_bound = [grad_norms[e] + noise_scalesl[e] for e in range(len(grad_norms))]
            plt.fill_between(epochs, lower_bound, upper_bound, alpha=0.2, label='DND Noise Range')
    
    plt.title('Gradient Noise Sensitivity by Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Gradient Norm')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"/u/yra7pn/llm_dp/gradient_noise_sensitivity_{task}.png")
    plt.close()

def plot_uncertainty_due_to_privacy(uncertainty_dict, task="sst2"):
    """
    Plot prediction uncertainty due to privacy.
    
    Args:
        uncertainty_dict (dict): Dictionary of prediction entropies per strategy.
        task (str): GLUE task name.
    """
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=[uncertainty_dict[strategy] for strategy in uncertainty_dict.keys()], 
                   palette='colorblind')
    plt.xticks(range(len(uncertainty_dict)), uncertainty_dict.keys())
    plt.title('Prediction Uncertainty Due to Privacy')
    plt.xlabel('Training Strategy')
    plt.ylabel('Prediction Entropy')
    plt.grid(True, axis='y')
    plt.savefig(f"/u/yra7pn/llm_dp/uncertainty_due_to_privacy_{task}.png")
    plt.close()

def plot_privacy_generalization_gap(task_results, task):
    """
    Plot privacy vs. generalization gap.
    
    Args:
        task_results (dict): Results for a specific task.
        task (str): GLUE task name.
    """
    strategies = list(task_results.keys())
    train_acc = [task_results[strategy]['train_accuracy'] for strategy in strategies]
    eval_acc = [task_results[strategy]['eval_accuracy'] for strategy in strategies]
    epsilons = [task_results[strategy]['epsilon'] if task_results[strategy]['epsilon'] is not None else None for strategy in strategies]
    
    generalization_gap = [train - eval for train, eval in zip(train_acc, eval_acc)]
    
    plt.figure(figsize=(10, 6))
    x = range(len(strategies))
    plt.bar([i - 0.2 for i in x], train_acc, width=0.2, label='Train Accuracy')
    plt.bar([i for i in x], eval_acc, width=0.2, label='Eval Accuracy')
    plt.bar([i + 0.2 for i in x], generalization_gap, width=0.2, label='Generalization Gap')
    
    for i, epsilon in enumerate(epsilons):
        if epsilon is not None:
            plt.text(i, eval_acc[i] + 0.01, f'ε={epsilon:.2f}', ha='center')
    
    plt.xticks(x, strategies)
    plt.title('Privacy vs. Generalization Gap')
    plt.xlabel('Training Strategy')
    plt.ylabel('Accuracy / Gap')
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig(f"/u/yra7pn/llm_dp/privacy_generalization_gap_{task}.png")
    plt.close()

def plot_all_visualizations(task, strategies, losses_dict, grad_norms_dict, uncertainty_dict, task_results, tasks, noise_multiplier):
    """
    Generate all visualizations for a task.
    
    Args:
        task (str): GLUE task name.
        strategies (list): List of training strategies.
        losses_dict (dict): Dictionary of losses per strategy.
        grad_norms_dict (dict): Dictionary of gradient norms per strategy.
        uncertainty_dict (dict): Dictionary of prediction entropies per strategy.
        task_results (dict): Results for the task.
        tasks (list): List of all GLUE tasks.
        noise_multiplier (float): Noise multiplier for DP strategies.
    """
    # Plot training curves
    plot_training_curves(strategies, losses_dict, grad_norms_dict, task)
    
    # Plot confusion matrices
    for strategy in strategies:
        plot_confusion_matrix(task_results[strategy]["cm"], strategy, task)
    
    # Plot additional visualizations
    plot_gradient_noise_sensitivity(grad_norms_dict, noise_multiplier, clip_norm=2.0, task=task)
    plot_uncertainty_due_to_privacy(uncertainty_dict, task=task)
    plot_privacy_generalization_gap(task_results, task)
    
    # Plot performance comparisons and privacy tradeoff for all tasks
    if task == tasks[-1]:  # Only plot once after all tasks are processed
        plot_performance_comparison({t: task_results for t in tasks}, tasks, "eval_accuracy")
        plot_performance_comparison({t: task_results for t in tasks}, tasks, "eval_f1")
        plot_privacy_performance_tradeoff({t: task_results for t in tasks}, tasks)
