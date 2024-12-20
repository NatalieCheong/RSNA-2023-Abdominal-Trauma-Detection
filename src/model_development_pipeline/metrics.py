import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, task_type):
        # Skip if no positive samples in batch
        if target.sum() == 0:
            return pred.sum() * 0

        # Define weights based on task type
        if task_type == 'bowel':
            weights = torch.tensor([1.0, 2.0])  # healthy=1, injury=2
        elif task_type == 'extravasation':
            weights = torch.tensor([1.0, 6.0])  # healthy=1, injury=6
        elif task_type == 'any_injury':
            weights = torch.tensor([1.0, 6.0])  # healthy=1, injury=6
        else:
            weights = torch.tensor([1.0, 1.0])  # default case

        weights = weights.to(pred.device)

        return F.cross_entropy(pred, target, weight=weights)


class MultiCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Weight pattern: healthy=1, low_grade=2, high_grade=4
        self.weights = {
            'kidney': torch.tensor([1.0, 2.0, 4.0]),
            'liver': torch.tensor([1.0, 2.0, 4.0]),
            'spleen': torch.tensor([1.0, 2.0, 4.0])
        }

    def forward(self, pred, target, organ_type):
        # Get the weights for the specific organ
        weights = self.weights[organ_type].to(pred.device)
        return F.cross_entropy(pred, target, weight=weights)
