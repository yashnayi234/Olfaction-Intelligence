"""
Graph Convolutional Network (GCN) Encoder.

Implements true graph convolution layers following Kipf & Welling (2017):
    H^(l+1) = σ(D^(-1/2) * A * D^(-1/2) * H^(l) * W^(l))

Uses pre-normalized adjacency matrices for efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from olfacnet import config


class GCNLayer(nn.Module):
    """Single Graph Convolutional Layer with batch normalization."""
    
    def __init__(self, in_features, out_features, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj):
        """
        Args:
            x: Node features (batch, num_nodes, in_features)
            adj: Normalized adjacency matrix (batch, num_nodes, num_nodes)
            
        Returns:
            Updated node features (batch, num_nodes, out_features)
        """
        # Graph convolution: A * X * W
        # adj: (B, N, N), x: (B, N, F_in)
        support = torch.bmm(adj, x)       # (B, N, F_in) — aggregate neighbor features
        out = self.linear(support)         # (B, N, F_out) — linear transform
        
        # BatchNorm expects (B, C, L), so transpose
        B, N, feat_dim = out.shape
        out = out.transpose(1, 2)          # (B, F_out, N)
        out = self.bn(out)
        out = out.transpose(1, 2)          # (B, N, F_out)
        
        out = F.relu(out)
        out = self.dropout(out)
        
        return out


class GCNEncoder(nn.Module):
    """
    Multi-layer GCN encoder with global mean pooling.
    
    Takes molecular graph (node features + adjacency) and produces a fixed-size
    molecular embedding vector.
    """
    
    def __init__(
        self,
        input_dim=None,
        hidden_dim=None,
        output_dim=None,
        num_layers=None,
        dropout=None,
    ):
        super().__init__()
        
        input_dim = input_dim or config.ATOM_FEATURE_DIM
        hidden_dim = hidden_dim or config.GNN_HIDDEN_DIM
        output_dim = output_dim or config.GNN_OUTPUT_DIM
        num_layers = num_layers or config.GNN_NUM_LAYERS
        dropout = dropout if dropout is not None else config.GNN_DROPOUT
        
        layers = []
        
        # Input layer
        layers.append(GCNLayer(input_dim, hidden_dim, dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(GCNLayer(hidden_dim, hidden_dim, dropout))
        
        # Output layer
        layers.append(GCNLayer(hidden_dim, output_dim, dropout=0.0))
        
        self.layers = nn.ModuleList(layers)
        self.output_dim = output_dim
    
    def forward(self, node_features, adj_matrix):
        """
        Args:
            node_features: (batch, max_atoms, feature_dim) — padded node features
            adj_matrix: (batch, max_atoms, max_atoms) — normalized padded adjacency
            
        Returns:
            Molecular embedding (batch, output_dim)
        """
        x = node_features
        
        # Apply GCN layers
        for layer in self.layers:
            x = layer(x, adj_matrix)
        
        # Global mean pooling (ignoring padding — zero features contribute zero)
        # Create mask from non-zero rows in original features
        mask = (node_features.abs().sum(dim=-1) > 0).float().unsqueeze(-1)  # (B, N, 1)
        x = x * mask  # Zero out padded nodes
        
        # Mean over actual atoms only
        num_atoms = mask.sum(dim=1).clamp(min=1)  # (B, 1) — avoid div by zero
        graph_embedding = x.sum(dim=1) / num_atoms  # (B, output_dim)
        
        return graph_embedding
