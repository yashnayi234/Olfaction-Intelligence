"""
Training loop with early stopping, learning rate scheduling, and gradient clipping.
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from olfacnet import config
from olfacnet.training.losses import FocalLoss
from olfacnet.training.metrics import compute_metrics, format_metrics


class Trainer:
    """
    Complete training loop for OlfacNet v2.
    
    Features:
        - Focal loss for class imbalance
        - AdamW optimizer with weight decay
        - ReduceLROnPlateau learning rate scheduling
        - Gradient clipping
        - Early stopping
        - Best model checkpointing
        - Training history logging
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader=None,
        learning_rate=None,
        weight_decay=None,
        epochs=None,
        patience=None,
        device=None,
        checkpoint_dir=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.device = device or config.DEVICE
        self.epochs = epochs or config.EPOCHS
        self.patience = patience or config.PATIENCE
        self.checkpoint_dir = checkpoint_dir or config.CHECKPOINTS_DIR
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Loss function
        self.criterion = FocalLoss()
        
        # Optimizer
        lr = learning_rate or config.LEARNING_RATE
        wd = weight_decay or config.WEIGHT_DECAY
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE,
            verbose=True,
        )
        
        # History
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_f1": [],
            "val_auc": [],
            "lr": [],
        }
        
        # Early stopping
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.best_epoch = 0
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_idx, (node_feat, adj, fp, labels) in enumerate(self.train_loader):
            node_feat = node_feat.to(self.device)
            adj = adj.to(self.device)
            fp = fp.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(node_feat, adj, fp)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), config.GRADIENT_CLIP_NORM
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Collect predictions for metrics
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.train_loader)
        
        # Compute training metrics
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        train_metrics = compute_metrics(all_labels, all_preds)
        
        return avg_loss, train_metrics
    
    @torch.no_grad()
    def validate(self, loader=None):
        """Validate the model."""
        if loader is None:
            loader = self.val_loader
        
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for node_feat, adj, fp, labels in loader:
            node_feat = node_feat.to(self.device)
            adj = adj.to(self.device)
            fp = fp.to(self.device)
            labels = labels.to(self.device)
            
            logits = self.model(node_feat, adj, fp)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        
        avg_loss = total_loss / len(loader)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        metrics = compute_metrics(all_labels, all_preds)
        
        return avg_loss, metrics, all_preds, all_labels
    
    def train(self):
        """
        Full training loop with early stopping.
        
        Returns:
            dict: Training history
        """
        print(f"\n{'='*60}")
        print(f"  OlfacNet v2 Training")
        print(f"  Device: {self.device}")
        print(f"  Parameters: {self.model.count_parameters():,}")
        print(f"  Epochs: {self.epochs} | Patience: {self.patience}")
        print(f"{'='*60}\n")
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        start_time = time.time()
        
        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_metrics = self.train_epoch()
            
            # Validate
            val_loss, val_metrics, _, _ = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_metrics["per_class_accuracy_mean"])
            self.history["val_acc"].append(val_metrics["per_class_accuracy_mean"])
            self.history["val_f1"].append(val_metrics["f1_macro"])
            self.history["val_auc"].append(val_metrics.get("auc_roc_macro", 0.0))
            self.history["lr"].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            
            # Print epoch summary
            print(
                f"  Epoch {epoch:3d}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val F1: {val_metrics['f1_macro']:.4f} | "
                f"Val AUC: {val_metrics.get('auc_roc_macro', 0):.4f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_epoch = epoch
                
                # Save best model
                self._save_checkpoint(epoch, val_loss, val_metrics)
                print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"\n  ⚡ Early stopping at epoch {epoch} (patience={self.patience})")
                    break
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"  Training Complete!")
        print(f"  Best epoch: {self.best_epoch} | Best val_loss: {self.best_val_loss:.4f}")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"{'='*60}\n")
        
        # Final evaluation on test set
        if self.test_loader:
            self._load_best_checkpoint()
            self._evaluate_test()
        
        # Save training history
        self._save_history()
        
        return self.history
    
    def _save_checkpoint(self, epoch, val_loss, val_metrics):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "val_metrics": val_metrics,
        }
        path = os.path.join(self.checkpoint_dir, "olfacnet_v2_best.pth")
        torch.save(checkpoint, path)
    
    def _load_best_checkpoint(self):
        """Load the best checkpoint."""
        path = os.path.join(self.checkpoint_dir, "olfacnet_v2_best.pth")
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"  Loaded best checkpoint from epoch {checkpoint['epoch']}")
    
    def _evaluate_test(self):
        """Evaluate on test set and print results."""
        print("\n  📊 Test Set Evaluation:")
        test_loss, test_metrics, _, _ = self.validate(self.test_loader)
        print(f"  Test Loss: {test_loss:.4f}")
        print(format_metrics(test_metrics))
        
        # Save test results
        results_dir = config.RESULTS_DIR
        os.makedirs(results_dir, exist_ok=True)
        
        results = {"test_loss": test_loss, **test_metrics}
        with open(os.path.join(results_dir, "test_results.json"), "w") as f:
            json.dump(results, f, indent=2)
    
    def _save_history(self):
        """Save training history to JSON."""
        results_dir = config.RESULTS_DIR
        os.makedirs(results_dir, exist_ok=True)
        
        with open(os.path.join(results_dir, "training_history.json"), "w") as f:
            json.dump(self.history, f, indent=2)
