import os
import torch
import seaborn as sns
import warnings

def set_device(gpu_id):
    """
    Set the GPU device and configure environment variables.
    
    Args:
        gpu_id (int): ID of the GPU to use.
    
    Returns:
        torch.device: Configured device (CUDA or CPU).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def initialize_environment():
    """
    Initialize environment settings, including Seaborn theme and warnings.
    """
    warnings.filterwarnings("ignore")
    cubehelix_palette = sns.color_palette("cubehelix", 6)
    sns.set_theme(style="whitegrid")
    sns.set_palette(cubehelix_palette)
