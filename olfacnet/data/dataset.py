"""
PyTorch Dataset and DataLoader creation for OlfacNet v2.

Uses pre-computed molecular graphs and generates Morgan fingerprints on-the-fly.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from olfacnet import config
from olfacnet.data.preprocessing import smiles_to_morgan_fp, pad_graph, normalize_adjacency
from olfacnet.data.utils import load_labels_dataframe


class OlfacDataset(Dataset):
    """
    PyTorch Dataset for OlfacNet training.
    
    Each sample returns:
        - node_features: (max_atoms, 1) padded node features
        - adj_matrix: (max_atoms, max_atoms) normalized padded adjacency matrix
        - fingerprint: (2048,) Morgan fingerprint vector
        - label: (num_classes,) multi-label binary vector
    """
    
    def __init__(self, smiles_list, labels_list, graphs_dir=None, max_atoms=None):
        """
        Args:
            smiles_list: List of SMILES strings
            labels_list: List of numpy arrays (multi-label binary vectors)
            graphs_dir: Directory containing pre-computed graph files
            max_atoms: Maximum atoms per molecule for padding
        """
        self.smiles_list = smiles_list
        self.labels_list = labels_list
        self.graphs_dir = graphs_dir or config.GRAPHS_DIR
        self.max_atoms = max_atoms or config.MAX_ATOMS
        
        # Build index mapping: SMILES → graph file index
        # The pre-computed graphs use sequential indices from the original dataset
        self._build_index()
    
    def _build_index(self):
        """Build mapping from SMILES to pre-computed graph file indices."""
        self.valid_indices = []
        self.graph_indices = []
        
        # Try to load the molecules_with_odors to get the mapping
        dataset_path = os.path.join(config.PROCESSED_DIR, "molecules_with_odors.csv")
        if os.path.exists(dataset_path):
            import pandas as pd
            df = pd.read_csv(dataset_path)
            if "IsomericSMILES" in df.columns:
                # Create SMILES → row index mapping from the original dataset
                smiles_to_orig_idx = {}
                for orig_idx, row in df.iterrows():
                    smiles_to_orig_idx[row["IsomericSMILES"]] = orig_idx
                
                for i, smiles in enumerate(self.smiles_list):
                    if smiles in smiles_to_orig_idx:
                        graph_idx = smiles_to_orig_idx[smiles]
                        node_path = os.path.join(self.graphs_dir, f"{graph_idx}_node_features.npy")
                        adj_path = os.path.join(self.graphs_dir, f"{graph_idx}_adj_matrix.npy")
                        if os.path.exists(node_path) and os.path.exists(adj_path):
                            self.valid_indices.append(i)
                            self.graph_indices.append(graph_idx)
        
        if not self.valid_indices:
            # Fallback: try to match by sequential index
            for i in range(len(self.smiles_list)):
                node_path = os.path.join(self.graphs_dir, f"{i}_node_features.npy")
                adj_path = os.path.join(self.graphs_dir, f"{i}_adj_matrix.npy")
                if os.path.exists(node_path) and os.path.exists(adj_path):
                    self.valid_indices.append(i)
                    self.graph_indices.append(i)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        data_idx = self.valid_indices[idx]
        graph_idx = self.graph_indices[idx]
        smiles = self.smiles_list[data_idx]
        label = self.labels_list[data_idx]
        
        # Load pre-computed graph
        node_features = np.load(
            os.path.join(self.graphs_dir, f"{graph_idx}_node_features.npy")
        )
        adj_matrix = np.load(
            os.path.join(self.graphs_dir, f"{graph_idx}_adj_matrix.npy")
        )
        
        # Pad graph to fixed size
        node_features, adj_matrix = pad_graph(node_features, adj_matrix, self.max_atoms)
        
        # Normalize adjacency matrix (GCN-style)
        adj_matrix = normalize_adjacency(adj_matrix)
        
        # Generate Morgan fingerprint
        fp = smiles_to_morgan_fp(smiles)
        if fp is None:
            fp = np.zeros(config.MORGAN_FP_BITS, dtype=np.float32)
        
        # Convert to tensors
        node_features = torch.FloatTensor(node_features)
        adj_matrix = torch.FloatTensor(adj_matrix)
        fingerprint = torch.FloatTensor(fp)
        label = torch.FloatTensor(label)
        
        return node_features, adj_matrix, fingerprint, label


def create_dataloaders(batch_size=None, num_workers=0):
    """
    Create train, validation, and test DataLoaders from the processed dataset.
    
    Args:
        batch_size: Batch size (default: from config)
        num_workers: Number of data loading workers
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, num_samples)
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    # Load labels
    print("Loading labels...")
    labels_df = load_labels_dataframe()
    print(f"  Loaded {len(labels_df)} samples with valid labels")
    
    smiles_list = labels_df["IsomericSMILES"].tolist()
    labels_list = labels_df["labels"].tolist()
    
    # Create full dataset
    full_dataset = OlfacDataset(smiles_list, labels_list)
    total = len(full_dataset)
    print(f"  Dataset has {total} valid samples with pre-computed graphs")
    
    # Split into train/val/test
    torch.manual_seed(config.RANDOM_SEED)
    train_size = int(config.TRAIN_SPLIT * total)
    val_size = int(config.VAL_SPLIT * total)
    test_size = total - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    print(f"  Split: {train_size} train / {val_size} val / {test_size} test")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader, total
