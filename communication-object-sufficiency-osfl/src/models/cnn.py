import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 9, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.proj = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(256, num_classes)
        self.feature_dim = 256

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = self.proj(x)
        return x

    def forward_head(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))

    def get_feature_dim(self) -> int:
        return self.feature_dim

    def set_linear_head(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        out_dim, in_dim = weight.shape
        if in_dim != self.feature_dim:
            raise ValueError(f"Head in_dim={in_dim} != feature_dim={self.feature_dim}")
        if self.classifier.out_features != out_dim:
            self.classifier = nn.Linear(self.feature_dim, out_dim).to(weight.device)
        with torch.no_grad():
            self.classifier.weight.copy_(weight)
            self.classifier.bias.copy_(bias)

    def encoder_parameters(self):
        for module in [self.features, self.proj]:
            yield from module.parameters()

    def head_parameters(self):
        yield from self.classifier.parameters()
