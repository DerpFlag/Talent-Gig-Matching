import torch
import torch.nn as nn


def build_bce_loss(positive_class_weight: float, device: torch.device) -> nn.Module:
    if positive_class_weight <= 0:
        raise ValueError("positive_class_weight must be > 0")
    pos_weight = torch.tensor([positive_class_weight], device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
