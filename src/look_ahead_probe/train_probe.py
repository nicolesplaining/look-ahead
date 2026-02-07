"""Training and evaluation functions for probes."""

from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data_loading import ActivationDataset
from .probe import FutureTokenProbe


def train_probe(
    probe: FutureTokenProbe,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    num_epochs: int,
    learning_rate: float,
    device: str = "cuda"
) -> dict:
    """Train probe and return history dict."""
    probe = probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        # Training
        probe.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for activations, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            activations = activations.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = probe(activations)
            loss = F.cross_entropy(logits, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            train_loss += loss.item()
            preds = logits.argmax(dim=-1)
            train_correct += (preds == targets).sum().item()
            train_total += len(targets)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # Validation
        if val_loader is not None:
            probe.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for activations, targets in val_loader:
                    activations = activations.to(device)
                    targets = targets.to(device)

                    logits = probe(activations)
                    loss = F.cross_entropy(logits, targets)

                    val_loss += loss.item()
                    preds = logits.argmax(dim=-1)
                    val_correct += (preds == targets).sum().item()
                    val_total += len(targets)

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
        else:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

    return history


def evaluate_probe(
    probe: FutureTokenProbe,
    test_loader: DataLoader,
    device: str = "cuda"
) -> dict:
    """Evaluate probe and return metrics dict."""
    probe.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for activations, targets in tqdm(test_loader, desc="Evaluating"):
            activations = activations.to(device)
            targets = targets.to(device)

            logits = probe(activations)
            loss = F.cross_entropy(logits, targets)

            test_loss += loss.item()
            preds = logits.argmax(dim=-1)
            test_correct += (preds == targets).sum().item()
            test_total += len(targets)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    test_loss /= len(test_loader)
    test_acc = test_correct / test_total

    # Top-k accuracy
    top5_correct = 0
    with torch.no_grad():
        for activations, targets in test_loader:
            activations = activations.to(device)
            targets = targets.to(device)

            logits = probe(activations)
            top5_preds = logits.topk(5, dim=-1).indices
            top5_correct += (top5_preds == targets.unsqueeze(-1)).any(dim=-1).sum().item()

    top5_acc = top5_correct / test_total

    return {
        "loss": test_loss,
        "accuracy": test_acc,
        "top5_accuracy": top5_acc,
        "predictions": all_preds,
        "targets": all_targets
    }
