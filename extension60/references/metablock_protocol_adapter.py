"""Faithful MetaBlock mechanism on the protocol-v2 ResNet50 representation."""
from __future__ import annotations

import torch
from torch import nn


class MetaBlock(nn.Module):
    """Author-code formula: sigmoid(tanh(V * f(U)) + g(U))."""

    def __init__(self, feature_groups: int, metadata_dim: int) -> None:
        super().__init__()
        self.fb = nn.Sequential(nn.Linear(metadata_dim, feature_groups), nn.BatchNorm1d(feature_groups))
        self.gb = nn.Sequential(nn.Linear(metadata_dim, feature_groups), nn.BatchNorm1d(feature_groups))

    def forward(self, visual: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        scale = self.fb(metadata)
        shift = self.gb(metadata)
        return torch.sigmoid(torch.tanh(visual * scale.unsqueeze(-1)) + shift.unsqueeze(-1))


class MetaBlockResNet50(nn.Module):
    """ResNet50 + the original 64-group MetaBlock + 256-unit reducer."""

    def __init__(self, num_classes: int = 6, metadata_dim: int = 5,
                 pretrained: bool = True, feature_groups: int = 64,
                 reducer_dim: int = 256, dropout: float = 0.5) -> None:
        super().__init__()
        if 2048 % feature_groups:
            raise ValueError("2048 ResNet features must divide evenly into MetaBlock groups")
        from torchvision.models import ResNet50_Weights, resnet50
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.metadata_dim = metadata_dim
        self.feature_groups = feature_groups
        self.metablock = MetaBlock(feature_groups, metadata_dim)
        self.reducer_block = nn.Sequential(
            nn.Linear(2048, reducer_dim),
            nn.BatchNorm1d(reducer_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.classifier = nn.Linear(reducer_dim, num_classes)

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        if metadata is None or metadata.ndim != 2 or metadata.shape[1] != self.metadata_dim:
            shape = None if metadata is None else tuple(metadata.shape)
            raise ValueError(f"Expected metadata [B,{self.metadata_dim}], got {shape}")
        visual = self.features(image)
        if visual.shape[1:] != (2048, 1, 1):
            raise RuntimeError(f"Unexpected ResNet50 pooled representation: {tuple(visual.shape)}")
        grouped = visual.view(visual.shape[0], self.feature_groups, -1)
        enhanced = self.metablock(grouped, metadata.float()).flatten(1)
        return self.classifier(self.reducer_block(enhanced))


def count_trainable(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
