"""
Inference pipeline for OlfacNet v2.

Provides a simple API for predicting odor profiles from SMILES strings.
"""

import os
import numpy as np
import torch

from olfacnet import config
from olfacnet.models.olfacnet import OlfacNetV2
from olfacnet.data.preprocessing import smiles_to_graph, smiles_to_morgan_fp, pad_graph, normalize_adjacency
from olfacnet.data.utils import load_odor_categories


class OlfacPredictor:
    """
    End-to-end SMILES → Odor prediction pipeline.
    
    Usage:
        predictor = OlfacPredictor("experiments/checkpoints/olfacnet_v2_best.pth")
        results = predictor.predict("CCO")  # Ethanol
        for odor, confidence in results:
            print(f"{odor}: {confidence:.2%}")
    """
    
    def __init__(self, checkpoint_path=None, device=None):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Torch device (default: auto-detect)
        """
        self.device = device or config.DEVICE
        self.odor_categories = load_odor_categories()
        
        # Initialize model
        self.model = OlfacNetV2()
        
        # Load checkpoint
        if checkpoint_path is None:
            checkpoint_path = os.path.join(config.CHECKPOINTS_DIR, "olfacnet_v2_best.pth")
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"✓ Model loaded from {checkpoint_path}")
        else:
            print(f"⚠ No checkpoint found at {checkpoint_path} — using untrained model")
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def predict(self, smiles, top_k=None, threshold=None):
        """
        Predict odor profile for a molecule.
        
        Args:
            smiles: SMILES string
            top_k: Number of top predictions to return (default: from config)
            threshold: Minimum confidence threshold (default: from config)
            
        Returns:
            list of (odor_name, confidence) tuples, sorted by confidence
        """
        top_k = top_k or config.TOP_K_PREDICTIONS
        threshold = threshold if threshold is not None else config.CONFIDENCE_THRESHOLD
        
        # Convert SMILES to graph
        node_features, adj_matrix = smiles_to_graph(smiles)
        if node_features is None:
            return []
        
        # Pad and normalize
        node_features, adj_matrix = pad_graph(node_features, adj_matrix)
        adj_matrix = normalize_adjacency(adj_matrix)
        
        # Convert to Morgan fingerprint
        fingerprint = smiles_to_morgan_fp(smiles)
        if fingerprint is None:
            fingerprint = np.zeros(config.MORGAN_FP_BITS, dtype=np.float32)
        
        # Convert to tensors and add batch dimension
        node_feat_tensor = torch.FloatTensor(node_features).unsqueeze(0).to(self.device)
        adj_tensor = torch.FloatTensor(adj_matrix).unsqueeze(0).to(self.device)
        fp_tensor = torch.FloatTensor(fingerprint).unsqueeze(0).to(self.device)
        
        # Predict
        probs = self.model.predict(node_feat_tensor, adj_tensor, fp_tensor)
        probs = probs.cpu().numpy().flatten()
        
        # Get top-K predictions above threshold
        results = []
        sorted_indices = np.argsort(probs)[::-1]
        
        for idx in sorted_indices[:top_k]:
            confidence = float(probs[idx])
            if confidence >= threshold and idx < len(self.odor_categories):
                odor_name = self.odor_categories[idx]
                results.append((odor_name, confidence))
        
        return results
    
    @torch.no_grad()
    def predict_full(self, smiles):
        """
        Get full probability distribution over all odor categories.
        
        Args:
            smiles: SMILES string
            
        Returns:
            dict mapping odor_name → probability
        """
        # Convert SMILES to graph
        node_features, adj_matrix = smiles_to_graph(smiles)
        if node_features is None:
            return {}
        
        # Pad and normalize
        node_features, adj_matrix = pad_graph(node_features, adj_matrix)
        adj_matrix = normalize_adjacency(adj_matrix)
        
        # Convert to Morgan fingerprint
        fingerprint = smiles_to_morgan_fp(smiles)
        if fingerprint is None:
            fingerprint = np.zeros(config.MORGAN_FP_BITS, dtype=np.float32)
        
        # Convert to tensors and add batch dimension
        node_feat_tensor = torch.FloatTensor(node_features).unsqueeze(0).to(self.device)
        adj_tensor = torch.FloatTensor(adj_matrix).unsqueeze(0).to(self.device)
        fp_tensor = torch.FloatTensor(fingerprint).unsqueeze(0).to(self.device)
        
        # Predict
        probs = self.model.predict(node_feat_tensor, adj_tensor, fp_tensor)
        probs = probs.cpu().numpy().flatten()
        
        return {
            self.odor_categories[i]: float(probs[i])
            for i in range(len(self.odor_categories))
            if i < len(probs)
        }
