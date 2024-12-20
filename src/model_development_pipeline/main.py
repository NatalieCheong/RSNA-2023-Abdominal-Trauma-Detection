import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from dataset import prepare_enhanced_datasets
from model import BinaryTraumaModel, MultiClassTraumaModel
from metrics import BinaryFocalLoss, MultiCELoss
import gc
from trainer import train_model, validate_model


def main_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Configuration
    config = {
        'batch_size': 16,
        'num_epochs': 15,
        'time_limit': 800,
        'binary_lr': 1e-4,
        'multi_lr': 2e-4,
        'weight_decay': 0.01,
    }

    # Get enhanced datasets
    train_dataset, val_dataset = prepare_enhanced_datasets()

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Initialize binary model (MobileNetV2)
    binary_model = BinaryTraumaModel().to(device)
    binary_criterion = BinaryFocalLoss()
    binary_optimizer = optim.AdamW(
        binary_model.parameters(),
        lr=config['binary_lr'],
        weight_decay=config['weight_decay']
    )
    binary_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        binary_optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )

    # Initialize multi-class model (ResNet34)
    multi_model = MultiClassTraumaModel().to(device)
    multi_criterion = MultiCELoss()
    multi_optimizer = optim.AdamW(
        multi_model.parameters(),
        lr=config['multi_lr'],
        weight_decay=config['weight_decay']
    )
    multi_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        multi_optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )

    # Training binary model
    print("Training binary classification model (MobileNetV2)...")
    best_binary_loss = float('inf')
    binary_patience = 0

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")

        train_loss = train_model(
            binary_model,
            train_loader,
            binary_criterion,
            binary_optimizer,
            config,
            device,
            'binary'
        )

        val_loss = validate_model(
            binary_model,
            val_loader,
            binary_criterion,
            device,
            'binary'
        )

        binary_scheduler.step(val_loss)

        if val_loss < best_binary_loss:
            best_binary_loss = val_loss
            torch.save(binary_model.state_dict(), 'best_binary_model.pth')
            print("Saved new best binary model!")
            binary_patience = 0
        else:
            binary_patience += 1
            if binary_patience >= 5:
                print("Early stopping triggered for binary model")
                break

    # Training multi-class model
    print("\nTraining multi-class model (ResNet34)...")
    best_multi_loss = float('inf')
    multi_patience = 0

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")

        train_loss = train_model(
            multi_model,
            train_loader,
            multi_criterion,
            multi_optimizer,
            config,
            device,
            'multi'
        )

        val_loss = validate_model(
            multi_model,
            val_loader,
            multi_criterion,
            device,
            'multi'
        )

        multi_scheduler.step(val_loss)

        if val_loss < best_multi_loss:
            best_multi_loss = val_loss
            torch.save(multi_model.state_dict(), 'best_multi_model.pth')
            print("Saved new best multi-class model!")
            multi_patience = 0
        else:
            multi_patience += 1
            if multi_patience >= 5:
                print("Early stopping triggered for multi-class model")
                break

    print("\nTraining completed!")
    print(f"Best binary model validation loss: {best_binary_loss:.4f}")
    print(f"Best multi-class model validation loss: {best_multi_loss:.4f}")

    # Memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main_training()
