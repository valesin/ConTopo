"""
ResNet18 backbone and end-to-end CE wrapper.

Ported from ``networks/modified_ResNet18.py`` — contrastive heads removed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """Basic residual block for ResNet18."""

    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet18(nn.Module):
    """Modified ResNet18 backbone for CIFAR-10 (stride-1 first conv)."""

    def __init__(self, in_channels: int = 3, emb_dim: int = 256) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = nn.Sequential(Block(64, 64), Block(64, 64))
        self.layer2 = nn.Sequential(Block(64, 128, stride=2), Block(128, 128))
        self.layer3 = nn.Sequential(Block(128, 256, stride=2), Block(256, 256))
        self.layer4 = nn.Sequential(Block(256, 512, stride=2), Block(512, 512))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, emb_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.flatten(1)
        return self.fc(out)


class LinearResNet18(nn.Module):
    """ResNet18 encoder + dropout + linear classifier.  Returns (embeddings, logits) when ``ret_emb=True``."""

    def __init__(
        self,
        emb_dim: int = 256,
        num_classes: int = 10,
        p_dropout: float = 0.5,
        use_dropout: bool = True,
        ret_emb: bool = False,
        head_bias: bool = True,
    ) -> None:
        super().__init__()
        self.ret_emb = ret_emb
        self.encoder = ResNet18(emb_dim=emb_dim)
        self.dropout = nn.Dropout(p_dropout) if use_dropout else nn.Identity()
        self.fc = nn.Linear(emb_dim, num_classes, bias=head_bias)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encoder(x)
        logits = self.fc(self.dropout(embeddings))
        return (embeddings, logits) if self.ret_emb else logits
