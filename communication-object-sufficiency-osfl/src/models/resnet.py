import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18ForMedMNIST(nn.Module):
    def __init__(self, num_classes: int = 9, in_channels: int = 3, dropout_p: float = 0.4):
        super().__init__()
        
        # 🌟 绝杀改造 1：使用 ImageNet 预训练权重
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # 🌟 绝杀改造 2：精细适配 conv1
        # 原版 ResNet18 的 conv1 是 7x7 的大卷积核，而我们这里用的是 3x3（为了适应 28x28 小图）
        # 我们不能随机初始化这个 3x3，而是要从预训练的 7x7 中“抠”出中间最核心的 3x3 权重！
        if in_channels != 3:
            # 针对单通道医疗图像（如 DermaMNIST 以外的部分数据集）
            original_conv1 = backbone.conv1
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            with torch.no_grad():
                # 抠出中心 3x3，并将 3 通道的权重求和浓缩到 1 个通道
                cropped_weight = original_conv1.weight[:, :, 2:5, 2:5]
                self.conv1.weight.copy_(cropped_weight.sum(dim=1, keepdim=True))
        else:
            # 针对三通道医疗图像（如 PathMNIST, DermaMNIST）
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            with torch.no_grad():
                # 抠出中心 3x3 保留原始特征提取能力
                self.conv1.weight.copy_(backbone.conv1.weight[:, :, 2:5, 2:5])

        # 去除 maxpool 以保留空间分辨率
        backbone.maxpool = nn.Identity()
        
        self.stem = nn.Sequential(self.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.dropout = nn.Dropout(dropout_p)
        self.feature_dim = backbone.fc.in_features
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return x

    def forward_head(self, features: torch.Tensor) -> torch.Tensor:
        features = self.dropout(features)
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
        modules = [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]
        for module in modules:
            yield from module.parameters()

    def head_parameters(self):
        yield from self.classifier.parameters()