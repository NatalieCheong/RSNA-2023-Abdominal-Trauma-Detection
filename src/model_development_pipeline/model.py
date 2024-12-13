import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class BinaryTraumaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('mobilenetv2_100', pretrained=True, in_chans=1)
        n_features = self.backbone.classifier.in_features

        self.shared = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(n_features),
            nn.Dropout(0.5),
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )

        # Match classifier names with target names
        self.classifiers = nn.ModuleDict({
            'bowel_injury': nn.Linear(256, 2),
            'extravasation_injury': nn.Linear(256, 2),
            'any_injury': nn.Linear(256, 2)
        })

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.shared(x)
        return {k: classifier(x) for k, classifier in self.classifiers.items()}


class MultiClassTraumaModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Use ResNet34 for more capacity
        self.backbone = timm.create_model('resnet34', pretrained=True, in_chans=1)
        n_features = self.backbone.fc.in_features

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)

        # Shared feature processing with more capacity
        self.shared_features = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(n_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            self.dropout1
        )

        # Separate classifiers for each organ
        self.classifiers = nn.ModuleDict({
            organ: nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                self.dropout2,
                nn.Linear(256, 3)
            ) for organ in ['kidney', 'liver', 'spleen']
        })

    def forward(self, x):
        features = self.backbone.forward_features(x)
        shared = self.shared_features(features)
        return {k: classifier(shared) for k, classifier in self.classifiers.items()}
