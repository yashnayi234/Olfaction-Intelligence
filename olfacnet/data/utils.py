"""
Data utility functions for label mapping, category loading, etc.
"""

import os
import ast
import numpy as np
import pandas as pd

from olfacnet import config


# ─── Odor Category List (105 categories from the dataset) ───────────────────
# This is populated at module load time from the dataset
_ODOR_CATEGORIES = None


def load_odor_categories(force_reload: bool = False):
    """
    Load the list of all unique odor categories from the dataset.
    
    Returns:
        list: Sorted list of odor category names
    """
    global _ODOR_CATEGORIES
    
    if _ODOR_CATEGORIES is not None and not force_reload:
        return _ODOR_CATEGORIES
    
    dataset_path = os.path.join(config.PROCESSED_DIR, "molecules_with_odors.csv")
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        if "Primary Odor" in df.columns:
            all_odors = set()
            for odors_str in df["Primary Odor"].dropna():
                for odor in str(odors_str).split(","):
                    odor = odor.strip()
                    if odor:
                        all_odors.add(odor)
            _ODOR_CATEGORIES = sorted(list(all_odors))
        else:
            _ODOR_CATEGORIES = []
    else:
        _ODOR_CATEGORIES = []
    
    return _ODOR_CATEGORIES


def get_label_mapping():
    """
    Get mapping from odor category name → index and vice versa.
    
    Returns:
        tuple: (odor_to_idx, idx_to_odor) dictionaries
    """
    categories = load_odor_categories()
    odor_to_idx = {odor: i for i, odor in enumerate(categories)}
    idx_to_odor = {i: odor for i, odor in enumerate(categories)}
    return odor_to_idx, idx_to_odor


def parse_label_vector(label_str: str) -> np.ndarray:
    """
    Parse a label string from labels.csv into a numpy array.
    
    The labels are stored as string representations of numpy arrays, e.g.:
    "[1. 0. 0. ... 0. 0. 0.]"
    
    Args:
        label_str: String representation of the label vector
        
    Returns:
        numpy array of shape (num_classes,)
    """
    # Clean up the string - handle multi-line numpy array format
    cleaned = label_str.strip()
    # Remove newlines and extra spaces
    cleaned = " ".join(cleaned.split())
    # Parse as numpy array
    cleaned = cleaned.replace("[", "").replace("]", "")
    values = [float(x) for x in cleaned.split()]
    return np.array(values, dtype=np.float32)


def load_labels_dataframe():
    """
    Load the labels CSV file with parsed label vectors.
    
    The labels.csv has a fixed format: each entry spans exactly 5 lines.
    Line 1: SMILES,"[val val val ... (24 values)
    Lines 2-4:  val val val ... (24 values each)
    Line 5:  val val val ... (9 values)]"
    
    Total: 24 + 24 + 24 + 24 + 9 = 105 values per entry.
    
    Returns:
        DataFrame with columns ['IsomericSMILES', 'labels'] where labels 
        is parsed into numpy arrays
    """
    labels_path = os.path.join(config.PROCESSED_DIR, "labels.csv")
    
    smiles_list = []
    labels_list = []
    
    with open(labels_path, "r") as f:
        lines = f.readlines()
    
    # Skip header (line 0), then process in chunks of 5
    data_lines = lines[1:]
    lines_per_entry = 5
    num_entries = len(data_lines) // lines_per_entry
    
    for i in range(num_entries):
        chunk = data_lines[i * lines_per_entry : (i + 1) * lines_per_entry]
        
        try:
            # First line contains SMILES,"[values...
            first_line = chunk[0].strip()
            parts = first_line.split(",", 1)
            if len(parts) != 2:
                continue
            
            smiles = parts[0].strip()
            
            # Join all 5 lines to get the full label string
            label_str = parts[1].strip()
            for j in range(1, len(chunk)):
                label_str += " " + chunk[j].strip()
            
            # Clean and parse
            label_vec = parse_label_vector(label_str.strip('"'))
            
            if len(label_vec) == config.NUM_CLASSES:
                smiles_list.append(smiles)
                labels_list.append(label_vec)
        except (ValueError, IndexError):
            continue
    
    return pd.DataFrame({
        "IsomericSMILES": smiles_list,
        "labels": labels_list
    })
