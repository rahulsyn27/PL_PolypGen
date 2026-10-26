import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple, Dict, Any, Optional

from src.utils.constants import NUM_CLASSES, INPUT_CHANNELS


class DecoupledModel(nn.Module):
    """
    Decoupled Model with explicit base feature extractor (body) and classifier (head).
    Used for FedRep (P1) and FedBN (P2).
    """
    def __init__(self, base: nn.Module, head: nn.Module):
        super().__init__()
        self.base = base
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if out.dim() > 2:
            out = torch.flatten(out, 1)
        return self.head(out)


def get_resnet18(num_classes: int = 2, in_channels: int = 3, pretrained: bool = False) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    if in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    in_features = model.fc.in_features
    
    # Split into body and head for DecoupledModel
    body = nn.Sequential(*list(model.children())[:-1])
    head = nn.Linear(in_features, num_classes)
    return DecoupledModel(base=body, head=head)


def get_resnet34(num_classes: int = 2, in_channels: int = 3, pretrained: bool = False) -> nn.Module:
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
    if in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    in_features = model.fc.in_features
    body = nn.Sequential(*list(model.children())[:-1])
    head = nn.Linear(in_features, num_classes)
    return DecoupledModel(base=body, head=head)


def get_simplecnn(num_classes: int = 2, in_channels: int = 3, pretrained: bool = False) -> nn.Module:
    """
    Wrap SimpleCNN in DecoupledModel so P1 (FedRep) and P2 (FedBN) layer
    routing works identically to the ResNet variants.
    """
    features = nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.AdaptiveAvgPool2d((1, 1))
    )
    head = nn.Linear(64, num_classes)
    return DecoupledModel(base=features, head=head)


MODELS = {
    "resnet18": get_resnet18,
    "resnet34": get_resnet34,
    "simplecnn": get_simplecnn,
}


def create_model(model_name: str = "resnet18", dataset_name: str = "polypgen", pretrained: bool = False) -> nn.Module:
    num_classes = NUM_CLASSES.get(dataset_name, 2)
    in_channels = INPUT_CHANNELS.get(dataset_name, 3)
    if model_name in MODELS:
        return MODELS[model_name](num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)
    return get_resnet18(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)
