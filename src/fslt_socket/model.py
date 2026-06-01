from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet18ClientSide(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.relu(self.layer1(x))
        out = self.layer2(residual)
        out += residual
        return F.relu(out)


class BaseBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        dim_change: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dim_change = dim_change

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.dim_change is not None:
            residual = self.dim_change(residual)
        out += residual
        return F.relu(out)


class ResNet18ServerSide(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.in_planes = 64
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
        )
        self.layer4 = self._make_layer(BaseBlock, 128, 2, stride=2)
        self.layer5 = self._make_layer(BaseBlock, 256, 2, stride=2)
        self.layer6 = self._make_layer(BaseBlock, 512, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BaseBlock.expansion, num_classes)

    def _make_layer(
        self,
        block: type[BaseBlock],
        planes: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        dim_change = None
        if stride != 1 or planes != self.in_planes:
            dim_change = nn.Sequential(
                nn.Conv2d(self.in_planes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers: list[nn.Module] = [block(self.in_planes, planes, stride, dim_change)]
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer3(x)
        out += x
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)
