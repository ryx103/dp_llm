import torch
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

def compute_epsilon(noise_multiplier, steps, delta=1e-5):
    """
    Compute epsilon for differential privacy.
    
    Args:
        noise_multiplier (float): Noise multiplier.
        steps (int): Number of training steps.
        delta (float): Delta parameter for DP.
    
    Returns:
        float: Epsilon value.
    """
    return (1.0 / noise_multiplier) ** 2 * steps / (2 * np.log(1.0 / delta))

def compute_grad_norm(model):
    """
    Compute the total gradient norm of the model.
    
    Args:
        model: PyTorch model.
    
    Returns:
        float: Total gradient norm.
    """
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.norm(2).item() ** 2
    return np.sqrt(total_norm)

def evaluate_model(model, dataloader, metric, device, task):
    """
    Evaluate model performance.
    
    Args:
        model: PyTorch model.
        dataloader: DataLoader for evaluation data.
        metric: Evaluation metric from the `evaluate` library.
        device: Device to evaluate on.
        task (str): GLUE task name.
    
    Returns:
        tuple: (metric_result, accuracy, f1, confusion_matrix)
    """
    model.eval()
    predictions, labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            batch_labels = batch["label"].numpy()
            predictions.extend(preds)
            labels.extend(batch_labels)
    
    accuracy = np.mean(np.array(predictions) == np.array(labels))
    f1 = f1_score(labels, predictions, average='binary' if task in ["sst2", "mrpc"] else 'macro')
    cm = confusion_matrix(labels, predictions)
    metric_result = metric.compute(predictions=predictions, references=labels)
    
    return metric_result, accuracy, f1, cm
