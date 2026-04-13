"""
CNN Model for Circuit Topology Classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CircuitClassifier(nn.Module):
    """
    A CNN architecture for classifying circuit gate images.
    Designed for small datasets with 7 classes.
    """

    def __init__(self, num_classes: int = 7, dropout_rate: float = 0.5):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Conv block 1: 224 -> 112
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Conv block 2: 112 -> 56
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Conv block 3: 56 -> 28
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Conv block 4: 28 -> 14
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        # Adaptive pooling -> 4x4
        x = self.adaptive_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x


class CircuitClassifierSmall(nn.Module):
    """
    A smaller CNN for faster training and less overfitting on small datasets.
    """

    def __init__(self, num_classes: int = 7, dropout_rate: float = 0.4):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Global average pooling
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResNetTransfer(nn.Module):
    """
    Transfer learning model using pretrained ResNet18.
    Supports full fine-tuning with differential learning rates.
    """

    def __init__(self, num_classes: int = 7, dropout_rate: float = 0.5, freeze_layers: bool = False):
        super().__init__()

        # Load pretrained ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Freeze early layers (optional)
        if freeze_layers:
            # Freeze all layers except layer4 and fc
            for name, param in self.backbone.named_parameters():
                if not name.startswith('layer4') and not name.startswith('fc'):
                    param.requires_grad = False

        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def get_parameter_groups(self, base_lr: float = 0.0001):
        """
        Get parameter groups with differential learning rates.
        Early layers get lower LR, later layers get higher LR.
        """
        # Early layers (conv1, bn1, layer1, layer2) - lowest LR
        early_params = []
        # Middle layers (layer3) - medium LR
        middle_params = []
        # Late layers (layer4) - higher LR
        late_params = []
        # Classifier (fc) - highest LR
        classifier_params = []

        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('fc'):
                classifier_params.append(param)
            elif name.startswith('layer4'):
                late_params.append(param)
            elif name.startswith('layer3'):
                middle_params.append(param)
            else:
                early_params.append(param)

        return [
            {'params': early_params, 'lr': base_lr * 0.1},
            {'params': middle_params, 'lr': base_lr * 0.5},
            {'params': late_params, 'lr': base_lr},
            {'params': classifier_params, 'lr': base_lr * 10},
        ]


class EfficientNetTransfer(nn.Module):
    """
    Transfer learning model using pretrained EfficientNet-B0.
    """

    def __init__(self, num_classes: int = 7, dropout_rate: float = 0.5, freeze_layers: bool = True):
        super().__init__()

        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Freeze early layers
        if freeze_layers:
            for name, param in self.backbone.named_parameters():
                if 'features.7' not in name and 'features.8' not in name and 'classifier' not in name:
                    param.requires_grad = False

        # Replace classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def get_model(model_name: str = "resnet", num_classes: int = 7, **kwargs) -> nn.Module:
    """
    Factory function to get a model by name.

    Args:
        model_name: "small", "standard", "resnet", or "efficientnet"
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to model constructor
    """
    available_models = {
        "small": CircuitClassifierSmall,
        "standard": CircuitClassifier,
        "resnet": ResNetTransfer,
        "efficientnet": EfficientNetTransfer,
    }

    if model_name not in available_models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(available_models.keys())}")

    return available_models[model_name](num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    # Test model
    model = get_model("small", num_classes=7)
    print(f"Model: {model.__class__.__name__}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    x = torch.randn(4, 3, 224, 224)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
