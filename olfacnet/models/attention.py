"""
Multi-Head Attention Fusion module.

Fuses embeddings from the GCN branch and the Fingerprint branch using
scaled dot-product multi-head attention. This allows the model to learn
which molecular representation is more informative for each prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from olfacnet import config


class MultiHeadAttentionFusion(nn.Module):
    """
    Multi-head cross-attention fusion for two embedding streams.
    
    Given embeddings from GCN (graph structure) and FP encoder (fingerprints),
    computes attention-weighted fusion.
    """
    
    def __init__(
        self,
        embed_dim_a=None,
        embed_dim_b=None,
        fusion_dim=None,
        num_heads=None,
        dropout=0.1,
    ):
        super().__init__()
        
        embed_dim_a = embed_dim_a or config.GNN_OUTPUT_DIM
        embed_dim_b = embed_dim_b or config.FP_OUTPUT_DIM
        fusion_dim = fusion_dim or config.ATTENTION_DIM
        num_heads = num_heads or config.ATTENTION_HEADS
        
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.head_dim = fusion_dim // num_heads
        
        assert fusion_dim % num_heads == 0, "fusion_dim must be divisible by num_heads"
        
        # Project both inputs to same dimension
        self.proj_a = nn.Linear(embed_dim_a, fusion_dim)
        self.proj_b = nn.Linear(embed_dim_b, fusion_dim)
        
        # Multi-head attention components
        self.W_q = nn.Linear(fusion_dim, fusion_dim)
        self.W_k = nn.Linear(fusion_dim, fusion_dim)
        self.W_v = nn.Linear(fusion_dim, fusion_dim)
        self.W_o = nn.Linear(fusion_dim, fusion_dim)
        
        self.layer_norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.output_dim = fusion_dim
    
    def forward(self, embed_a, embed_b):
        """
        Args:
            embed_a: (batch, embed_dim_a) — GCN graph embedding
            embed_b: (batch, embed_dim_b) — Fingerprint embedding
            
        Returns:
            Fused embedding (batch, fusion_dim)
        """
        B = embed_a.size(0)
        
        # Project to common dimension
        a = self.proj_a(embed_a)  # (B, fusion_dim)
        b = self.proj_b(embed_b)  # (B, fusion_dim)
        
        # Stack as sequence of 2 tokens: [graph_embed, fp_embed]
        seq = torch.stack([a, b], dim=1)  # (B, 2, fusion_dim)
        
        # Multi-head self-attention over the 2 tokens
        Q = self.W_q(seq)  # (B, 2, fusion_dim)
        K = self.W_k(seq)
        V = self.W_v(seq)
        
        # Reshape for multi-head: (B, num_heads, 2, head_dim)
        Q = Q.view(B, 2, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, 2, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, 2, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, heads, 2, 2)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)  # (B, heads, 2, head_dim)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, 2, self.fusion_dim)
        attn_output = self.W_o(attn_output)
        
        # Residual connection + layer norm
        attn_output = self.layer_norm(attn_output + seq)
        
        # Pool over the 2 tokens (mean)
        fused = attn_output.mean(dim=1)  # (B, fusion_dim)
        
        return fused
