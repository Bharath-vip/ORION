import torch
import torch.nn.functional as F

def get_drw_weights(class_frequencies, epoch, drw_epoch):
    """
    Returns uniform weights before DRW epoch, and inverse frequency weights after.
    """
    if epoch < drw_epoch:
        return torch.ones_like(class_frequencies, dtype=torch.float32)
    else:
        weights = 1.0 / class_frequencies
        # Normalize weights to keep loss magnitude stable
        weights = weights / weights.sum() * len(class_frequencies)
        return weights
