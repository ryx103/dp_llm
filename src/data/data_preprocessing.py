from datasets import load_dataset
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader
import numpy as np

def preprocess_data(dataset, tokenizer, task, max_length=128):
    """
    Preprocess dataset for a specific GLUE task.
    
    Args:
        dataset: Dataset to preprocess.
        tokenizer: DistilBertTokenizer instance.
        task (str): GLUE task name (sst2, cola, mrpc).
        max_length (int): Maximum sequence length.
    
    Returns:
        Dataset: Preprocessed dataset.
    """
    def preprocess_function(examples):
        if task == "mrpc":
            return tokenizer(
                examples["sentence1"],
                examples["sentence2"],
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
        elif task in ["sst2", "cola"]:
            return tokenizer(
                examples["sentence"],
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
        else:
            raise ValueError(f"Unsupported task: {task}")
    
    dataset = dataset.map(preprocess_function, batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return dataset

def load_and_preprocess_data(task, train_size, eval_size, batch_size):
    """
    Load and preprocess data for training and evaluation.
    
    Args:
        task (str): GLUE task name.
        train_size (int): Size of training dataset.
        eval_size (int): Size of evaluation dataset.
        batch_size (int): Batch size for DataLoader.
    
    Returns:
        tuple: (train_loader, eval_loader, tokenizer)
    """
    # Load dataset
    dataset = load_dataset("glue", task)
    actual_eval_size = min(eval_size, len(dataset["validation"]))

    # Select subsets
    train_dataset = dataset["train"].shuffle(seed=42).select(range(train_size))
    eval_dataset = dataset["validation"].select(range(actual_eval_size))
    
    # Print label distributions
    print(f"Train label distribution: {np.bincount(train_dataset['label'])}")
    print(f"Eval label distribution: {np.bincount(eval_dataset['label'])}")
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Preprocess datasets
    train_dataset = preprocess_data(train_dataset, tokenizer, task)
    eval_dataset = preprocess_data(eval_dataset, tokenizer, task)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
    
    return train_loader, eval_loader, tokenizer
