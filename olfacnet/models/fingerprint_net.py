"""
Fingerprint Encoder — encodes Morgan fingerprints into dense embeddings.

Replaces the original SpectrumCNN with a more informative molecular 
representation. Morgan fingerprints capture circular substructural patterns 
and are the gold standard in computational chemistry.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from olfacnet import config


class FingerprintEncoder(nn.Module):
    """
    MLP encoder for Morgan fingerprint vectors.
    
    Architecture:
        Input (2048) → FC (512) → BN → ReLU → Dropout
                     → FC (256) → BN → ReLU → Dropout  
                     → FC (128) — output embedding
    """
    
    def __init__(
        self,
        input_dim=None,
        hidden_dim=None,
        output_dim=None,
        dropout=0.3,
    ):
        super().__init__()
        
        input_dim = input_dim or config.MORGAN_FP_BITS
        hidden_dim = hidden_dim or config.FP_HIDDEN_DIM
        output_dim = output_dim or config.FP_OUTPUT_DIM
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, output_dim),
        )
        
        self.output_dim = output_dim
    
    def forward(self, fingerprint):
        """
        Args:
            fingerprint: (batch, fp_bits) Morgan fingerprint vectors
            
        Returns:
            Fingerprint embedding (batch, output_dim)
        """
        return self.encoder(fingerprint)
