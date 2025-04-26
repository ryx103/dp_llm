import torch
from torch.optim import AdamW
from transformers import DistilBertForSequenceClassification
from scipy.stats import entropy
from src.evaluation.metrics import evaluate_model, compute_epsilon, compute_grad_norm

def train_model(model, train_loader, eval_loader, optimizer, strategy, device, epochs=10, noise_multiplier=0.1, task="sst2"):
    """
    Train a model with specified strategy.
    
    Args:
        model: DistilBERT model.
        train_loader: DataLoader for training data.
        eval_loader: DataLoader for evaluation data.
        optimizer: Optimizer instance.
        strategy (str): Training strategy (SGD, DP-SGD, DND, Adaptive Clipping).
        device: Device to train on.
        epochs (int): Number of training epochs.
        noise_multiplier (float): Noise multiplier for DP strategies.
        task (str): GLUE task name.
    
    Returns:
        tuple: (losses, grad_norms, epsilon)
    """
    losses, grad_norms = [], []
    steps = 0
    clip_norm = 2.0
    
    for epoch in range(epochs):
        model.train()
        total_loss, total_grad_norm = 0, 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            labels = batch["label"].to(device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            if strategy == "DP-SGD":
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad += torch.normal(0, noise_multiplier * clip_norm / len(train_loader.dataset), param.grad.shape).to(device)
                total_grad_norm += grad_norm.item()
            elif strategy == "DND":
                dynamic_noise = noise_multiplier * (0.95 ** epoch)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad += torch.normal(0, dynamic_noise * clip_norm / len(train_loader.dataset), param.grad.shape).to(device)
                total_grad_norm += grad_norm.item()
            elif strategy == "Adaptive Clipping":
                grad_norm = compute_grad_norm(model)
                if grad_norm > clip_norm:
                    clip_norm = min(grad_norm * 1.1, 5.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                total_grad_norm += grad_norm
            else:  # SGD
                grad_norm = compute_grad_norm(model)
                total_grad_norm += grad_norm
            
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        
        avg_loss = total_loss / len(train_loader)
        avg_grad_norm = total_grad_norm / len(train_loader)
        losses.append(avg_loss)
        grad_norms.append(avg_grad_norm)
        print(f"{strategy} Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, Avg Grad Norm: {avg_grad_norm:.4f}")
    
    epsilon = compute_epsilon(noise_multiplier, steps) if strategy in ["DP-SGD", "DND"] else None
    return losses, grad_norms, epsilon

def train_and_evaluate(task, train_loader, eval_loader, strategies, device, tokenizer, epochs, lr, noise_multiplier):
    """
    Train and evaluate models for all strategies.
    
    Args:
        task (str): GLUE task name.
        train_loader: DataLoader for training data.
        eval_loader: DataLoader for evaluation data.
        strategies (list): List of training strategies.
        device: Device to train on.
        tokenizer: DistilBertTokenizer instance.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        noise_multiplier (float): Noise multiplier for DP strategies.
    
    Returns:
        tuple: (task_results, losses_dict, grad_norms_dict, uncertainty_dict)
    """
    from evaluate import load
    
    task_results = {}
    losses_dict = {}
    grad_norms_dict = {}
    uncertainty_dict = {strategy: [] for strategy in strategies}
    
    for strategy in strategies:
        print(f"\nRunning {strategy} on {task}...")
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)
        for param in model.distilbert.parameters():
            param.requires_grad = False
        optimizer = AdamW(model.parameters(), lr=lr)
        
        losses, grad_norms, epsilon = train_model(
            model, train_loader, eval_loader, optimizer, strategy, device, epochs, noise_multiplier, task
        )
        losses_dict[strategy] = losses
        grad_norms_dict[strategy] = grad_norms
        
        metric = load("glue", task)
        train_metric, train_accuracy, train_f1, train_cm = evaluate_model(model, train_loader, metric, device, task)
        eval_metric, eval_accuracy, eval_f1, eval_cm = evaluate_model(model, eval_loader, metric, device, task)
        
        task_results[strategy] = {
            "train_accuracy": train_accuracy,
            "eval_accuracy": eval_accuracy,
            "train_f1": train_f1,
            "eval_f1": eval_f1,
            "epsilon": epsilon,
            "cm": eval_cm
        }
        
        # Compute prediction entropy for uncertainty
        model.eval()
        with torch.no_grad():
            for batch in eval_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                for prob in probs:
                    uncertainty_dict[strategy].append(entropy(prob))
        
        print(f"{strategy} Train Accuracy: {train_accuracy:.4f}, Eval Accuracy: {eval_accuracy:.4f}, "
              f"F1: {eval_f1:.4f}, Epsilon: {epsilon if epsilon else 'N/A'}")
    
    return task_results, losses_dict, grad_norms_dict, uncertainty_dict
