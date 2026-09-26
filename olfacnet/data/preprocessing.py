"""
Preprocessing utilities for converting molecular data to model-ready tensors.

- SMILES → molecular graph (adjacency matrix + node features)
- SMILES → Morgan fingerprint (2048-bit)
"""

import os
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

from olfacnet import config


def smiles_to_graph(smiles: str, idx: int = 0, output_dir: str = None):
    """
    Convert a SMILES string to graph representation (node features + adjacency matrix).
    
    Args:
        smiles: SMILES string of the molecule
        idx: Index for saving/caching
        output_dir: Directory to save graph files (optional)
        
    Returns:
        tuple: (node_features, adj_matrix) as numpy arrays, or (None, None) if invalid
    """
    if not HAS_RDKIT:
        raise ImportError("RDKit is required for SMILES→Graph conversion. Install with: pip install rdkit")
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    
    # Node features: atomic number (normalized)
    atoms = mol.GetAtoms()
    num_atoms = len(atoms)
    node_features = np.array([[atom.GetAtomicNum()] for atom in atoms], dtype=np.float32)
    
    # Adjacency matrix
    adj_matrix = np.zeros((num_atoms, num_atoms), dtype=np.float32)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj_matrix[i][j] = 1.0
        adj_matrix[j][i] = 1.0
    
    # Add self-loops
    adj_matrix += np.eye(num_atoms, dtype=np.float32)
    
    # Save if output_dir specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f"{idx}_node_features.npy"), node_features)
        np.save(os.path.join(output_dir, f"{idx}_adj_matrix.npy"), adj_matrix)
    
    return node_features, adj_matrix


def smiles_to_morgan_fp(smiles: str, n_bits: int = None, radius: int = None):
    """
    Convert a SMILES string to a Morgan fingerprint vector.
    
    Morgan fingerprints encode circular substructural patterns — the industry
    standard for molecular representation in cheminformatics.
    
    Args:
        smiles: SMILES string of the molecule
        n_bits: Length of the fingerprint vector (default: from config)
        radius: Radius of the Morgan fingerprint (default: from config)
        
    Returns:
        numpy array of shape (n_bits,) with float32 values, or None if invalid
    """
    if not HAS_RDKIT:
        raise ImportError("RDKit is required for Morgan fingerprints. Install with: pip install rdkit")
    
    if n_bits is None:
        n_bits = config.MORGAN_FP_BITS
    if radius is None:
        radius = config.MORGAN_RADIUS
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    try:
        from rdkit.Chem import rdFingerprintGenerator
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = gen.GetFingerprintAsNumPy(mol)
        return fp.astype(np.float32)
    except (ImportError, AttributeError):
        # Fallback for older rdkit versions
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp, dtype=np.float32)


def pad_graph(node_features, adj_matrix, max_atoms: int = None):
    """
    Pad graph tensors to a fixed size for batching.
    
    Args:
        node_features: (N, F) node feature matrix
        adj_matrix: (N, N) adjacency matrix
        max_atoms: Maximum number of atoms (default: from config)
        
    Returns:
        tuple: (padded_node_features, padded_adj_matrix) with shapes (max_atoms, F) and (max_atoms, max_atoms)
    """
    if max_atoms is None:
        max_atoms = config.MAX_ATOMS
    
    num_atoms = node_features.shape[0]
    feat_dim = node_features.shape[1]
    
    if num_atoms > max_atoms:
        # Truncate if molecule is too large
        node_features = node_features[:max_atoms]
        adj_matrix = adj_matrix[:max_atoms, :max_atoms]
        num_atoms = max_atoms
    
    # Pad node features
    padded_features = np.zeros((max_atoms, feat_dim), dtype=np.float32)
    padded_features[:num_atoms] = node_features
    
    # Pad adjacency matrix
    padded_adj = np.zeros((max_atoms, max_atoms), dtype=np.float32)
    padded_adj[:num_atoms, :num_atoms] = adj_matrix
    
    return padded_features, padded_adj


def normalize_adjacency(adj_matrix):
    """
    Symmetric normalization of adjacency matrix: D^(-1/2) * A * D^(-1/2)
    This is the standard GCN normalization from Kipf & Welling (2017).
    
    Args:
        adj_matrix: (N, N) adjacency matrix (with self-loops)
        
    Returns:
        Normalized adjacency matrix
    """
    degree = np.sum(adj_matrix, axis=1)
    # Avoid division by zero
    degree_inv_sqrt = np.where(degree > 0, np.power(degree, -0.5), 0.0)
    D_inv_sqrt = np.diag(degree_inv_sqrt)
    return D_inv_sqrt @ adj_matrix @ D_inv_sqrt
