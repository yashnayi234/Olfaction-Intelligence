"""
CLI entry point for evaluating OlfacNet v2.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint experiments/checkpoints/olfacnet_v2_best.pth
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from olfacnet import config
from olfacnet.data.dataset import create_dataloaders
from olfacnet.models.olfacnet import OlfacNetV2
from olfacnet.training.metrics import compute_metrics, format_metrics
from olfacnet.training.losses import FocalLoss
from olfacnet.data.utils import load_odor_categories


def main():
    parser = argparse.ArgumentParser(description="Evaluate OlfacNet v2")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(config.CHECKPOINTS_DIR, "olfacnet_v2_best.pth"),
        help="Path to model checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()
    
    device = config.DEVICE
    print(f"\n📊 OlfacNet v2 — Model Evaluation")
    print(f"{'─'*50}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device:     {device}")
    print(f"{'─'*50}\n")
    
    # Load model
    model = OlfacNetV2()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  ✓ Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Load data
    _, _, test_loader, _ = create_dataloaders(batch_size=args.batch_size)
    
    # Evaluate
    criterion = FocalLoss()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for node_feat, adj, fp, labels in test_loader:
            node_feat = node_feat.to(device)
            adj = adj.to(device)
            fp = fp.to(device)
            labels = labels.to(device)
            
            logits = model(node_feat, adj, fp)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    metrics = compute_metrics(all_labels, all_preds)
    metrics["test_loss"] = avg_loss
    
    print(f"\n  Test Loss: {avg_loss:.4f}")
    print(format_metrics(metrics))
    
    # Save results
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(config.RESULTS_DIR, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Results saved to: {results_path}")
    
    # Per-class analysis for top/bottom categories
    categories = load_odor_categories()
    per_class = []
    for i in range(all_labels.shape[1]):
        pos_count = all_labels[:, i].sum()
        if pos_count > 0 and i < len(categories):
            pred_binary = (all_preds[:, i] >= 0.5).astype(float)
            correct = (pred_binary == all_labels[:, i]).mean()
            per_class.append((categories[i], correct, int(pos_count)))
    
    per_class.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n  🏆 Top 10 Best Predicted Categories:")
    for name, acc, count in per_class[:10]:
        print(f"     {name:30s} | Acc: {acc:.4f} | Samples: {count}")
    
    print(f"\n  ⚠ Bottom 10 Categories:")
    for name, acc, count in per_class[-10:]:
        print(f"     {name:30s} | Acc: {acc:.4f} | Samples: {count}")


if __name__ == "__main__":
    main()
