"""
CLI entry point for training OlfacNet v2.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 50 --batch-size 32 --lr 0.0005
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from olfacnet import config
from olfacnet.data.dataset import create_dataloaders
from olfacnet.models.olfacnet import OlfacNetV2
from olfacnet.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train OlfacNet v2")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--patience", type=int, default=config.PATIENCE, help="Early stopping patience")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()
    
    # Override config if needed
    if args.device:
        device = torch.device(args.device)
    else:
        device = config.DEVICE
    
    print(f"\n🧪 OlfacNet v2 — Olfactory Intelligence Training")
    print(f"{'─'*50}")
    print(f"  Device:     {device}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Patience:   {args.patience}")
    print(f"{'─'*50}\n")
    
    # Create data loaders
    print("📂 Loading dataset...")
    train_loader, val_loader, test_loader, total = create_dataloaders(
        batch_size=args.batch_size
    )
    
    # Create model
    print("\n🏗  Building OlfacNet v2...")
    model = OlfacNetV2()
    print(f"  Total parameters: {model.count_parameters():,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        device=device,
    )
    
    # Train!
    print("\n🚀 Starting training...")
    history = trainer.train()
    
    print("\n✅ Training complete! Checkpoint saved to:")
    print(f"   {os.path.join(config.CHECKPOINTS_DIR, 'olfacnet_v2_best.pth')}")


if __name__ == "__main__":
    main()
