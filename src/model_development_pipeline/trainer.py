import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, auc, accuracy_score, roc_curve, classification_report, f1_score
import time

def train_model(model, train_loader, criterion, optimizer, config, device, model_type):
    """Streamlined training function with minimal output"""
    model.train()
    total_loss = 0
    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(train_loader):
        # Check time limit
        if time.time() - start_time > config['time_limit']:
            break

        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        optimizer.zero_grad()
        outputs = model(images)

        # Calculate loss based on model type
        if model_type == 'binary':
            losses = [criterion(outputs[k], targets[k], k) for k in outputs]
        else:  # multi-class
            losses = [criterion(outputs[k], targets[k], k) for k in outputs]

        loss = sum(losses) / len(losses)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    final_avg_loss = total_loss / (batch_idx + 1)
    print(f"Training Loss: {final_avg_loss:.4f}")
    return final_avg_loss

def validate_model(model, val_loader, criterion, device, model_type):
    """Streamlined validation function with minimal output"""
    model.eval()
    total_loss = 0
    all_predictions = {}
    all_targets = {}

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            outputs = model(images)

            # Calculate loss
            if model_type == 'binary':
                losses = [criterion(outputs[k], targets[k], k) for k in outputs]
            else:
                losses = [criterion(outputs[k], targets[k], k) for k in outputs]
            loss = sum(losses) / len(losses)
            total_loss += loss.item()

            # Store predictions and targets
            for k in outputs:
                if k not in all_predictions:
                    all_predictions[k] = []
                    all_targets[k] = []

                if model_type == 'binary':
                    probs = torch.softmax(outputs[k], dim=1)
                    all_predictions[k].extend(probs[:, 1].cpu().numpy())
                else:
                    preds = outputs[k].argmax(dim=1)
                    all_predictions[k].extend(preds.cpu().numpy())
                all_targets[k].extend(targets[k].cpu().numpy())

    # Calculate and print metrics
    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}\n")
    print("Validation Metrics:")

    for k in all_predictions:
        preds = np.array(all_predictions[k])
        targets = np.array(all_targets[k])
        accuracy = accuracy_score(targets, (preds > 0.5).astype(int) if model_type == 'binary' else preds)

        if model_type == 'binary':
            try:
                auc = roc_auc_score(targets, preds)
                print(f"{k}: Accuracy = {accuracy:.4f}, AUC = {auc:.4f}")
            except:
                print(f"{k}: Accuracy = {accuracy:.4f}")
        else:
            print(f"{k}: Accuracy = {accuracy:.4f}")

    return avg_loss
