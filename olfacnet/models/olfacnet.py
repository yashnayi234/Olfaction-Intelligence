"""
OlfacNet v2 — The complete olfactory prediction model.

Architecture:
    SMILES → [GCN Branch] ──┐
                             ├── Multi-Head Attention Fusion → Classifier → σ(105)
    SMILES → [FP Branch]  ──┘

Predicts 105 odor categories from molecular structure using multi-label classification.
"""

import torch
import torch.nn as nn

from olfacnet import config
from olfacnet.models.gnn import GCNEncoder
from olfacnet.models.fingerprint_net import FingerprintEncoder
from olfacnet.models.attention import MultiHeadAttentionFusion


class OlfacNetV2(nn.Module):
    """
    OlfacNet v2: Multi-modal olfactory prediction network.
    
    Combines graph neural network analysis of molecular structure with
    Morgan fingerprint encoding, fused via multi-head attention.
    """
    
    def __init__(
        self,
        num_classes=None,
        gcn_input_dim=None,
        gcn_hidden_dim=None,
        gcn_output_dim=None,
        gcn_num_layers=None,
        gcn_dropout=None,
        fp_input_dim=None,
        fp_hidden_dim=None,
        fp_output_dim=None,
        attention_dim=None,
        attention_heads=None,
        classifier_hidden=None,
        classifier_dropout=None,
    ):
        super().__init__()
        
        num_classes = num_classes or config.NUM_CLASSES
        classifier_hidden = classifier_hidden or config.CLASSIFIER_HIDDEN
        classifier_dropout = classifier_dropout if classifier_dropout is not None else config.CLASSIFIER_DROPOUT
        attention_dim = attention_dim or config.ATTENTION_DIM
        
        # ─── Branch 1: Graph Convolutional Network ───────────────────────
        self.gnn = GCNEncoder(
            input_dim=gcn_input_dim,
            hidden_dim=gcn_hidden_dim,
            output_dim=gcn_output_dim,
            num_layers=gcn_num_layers,
            dropout=gcn_dropout,
        )
        
        # ─── Branch 2: Morgan Fingerprint Encoder ────────────────────────
        self.fp_encoder = FingerprintEncoder(
            input_dim=fp_input_dim,
            hidden_dim=fp_hidden_dim,
            output_dim=fp_output_dim,
        )
        
        # ─── Attention Fusion ────────────────────────────────────────────
        self.fusion = MultiHeadAttentionFusion(
            embed_dim_a=self.gnn.output_dim,
            embed_dim_b=self.fp_encoder.output_dim,
            fusion_dim=attention_dim,
            num_heads=attention_heads,
        )
        
        # ─── Classification Head ─────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion.output_dim, classifier_hidden),
            nn.BatchNorm1d(classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(classifier_dropout),
            
            nn.Linear(classifier_hidden, classifier_hidden // 2),
            nn.BatchNorm1d(classifier_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(classifier_dropout * 0.5),
            
            nn.Linear(classifier_hidden // 2, num_classes),
        )
    
    def forward(self, node_features, adj_matrix, fingerprint):
        """
        Args:
            node_features: (batch, max_atoms, atom_feat_dim)
            adj_matrix: (batch, max_atoms, max_atoms)
            fingerprint: (batch, fp_bits)
            
        Returns:
            Prediction logits (batch, num_classes) — apply sigmoid for probabilities
        """
        # Branch 1: Graph embedding
        graph_embed = self.gnn(node_features, adj_matrix)
        
        # Branch 2: Fingerprint embedding
        fp_embed = self.fp_encoder(fingerprint)
        
        # Fuse via attention
        fused = self.fusion(graph_embed, fp_embed)
        
        # Classify
        logits = self.classifier(fused)
        
        return logits
    
    def predict(self, node_features, adj_matrix, fingerprint):
        """
        Run prediction with sigmoid activation.
        
        Returns:
            Probabilities (batch, num_classes) in [0, 1]
        """
        logits = self.forward(node_features, adj_matrix, fingerprint)
        return torch.sigmoid(logits)
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
