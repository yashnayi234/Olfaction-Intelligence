"""
Custom loss functions for multi-label odor classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from olfacnet import config


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    
    Addresses class imbalance by down-weighting easy examples and focusing
    training on hard, misclassified ones. Originally from Lin et al. (2017).
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: Weighting factor for the rare class (default: 0.25)
        gamma: Focusing parameter — higher values focus more on hard examples (default: 2.0)
    """
    
    def __init__(self, alpha=None, gamma=None, reduction="mean"):
        super().__init__()
        self.alpha = alpha if alpha is not None else config.FOCAL_ALPHA
        self.gamma = gamma if gamma is not None else config.FOCAL_GAMMA
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (batch, num_classes) — raw model output (before sigmoid)
            targets: (batch, num_classes) — binary multi-label targets
            
        Returns:
            Focal loss scalar
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Binary cross entropy per element
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        
        # Compute p_t (probability of the true class)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Compute alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal modulation factor
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)
        
        # Final focal loss
        loss = focal_weight * bce
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
