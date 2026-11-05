"""HFPA-Net implementation for PAD-UFES-20 multimodal classification.

Shapes:
    image: [B, 3, 224, 224]
    meta:  [B, meta_in_dim]
    logits: [B, num_classes]
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module_wrappers import AFPN, B2C_BFAM, CBM, DA_Block, DWT_2D, HLFDB_LOAD_ERROR, MSCBLayer, diff_moudel


def _conv_bn_relu(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ImageBackbone(nn.Module):
    """Extract four image feature maps from a torchvision backbone."""

    def __init__(self, backbone_name: str = "resnet50", pretrained: bool = True, prefer_timm: bool = False):
        super().__init__()
        self.backbone_name = backbone_name
        self.prefer_timm = prefer_timm
        self.pretrained = pretrained
        self._backend = None
        self.feature_dims: List[int] = []
        self._build_backbone()

    def _build_backbone(self) -> None:
        if self.backbone_name == "resnet50":
            self._build_resnet50()
            return
        if self.backbone_name == "vgg16":
            self._build_vgg16()
            return
        if self.backbone_name == "mobilenet_v2":
            self._build_mobilenet_v2()
            return
        raise ValueError(f"Unsupported backbone: {self.backbone_name}")

    def _build_resnet50(self) -> None:
        try:
            from torchvision.models import ResNet50_Weights, resnet50

            weights = ResNet50_Weights.IMAGENET1K_V2 if self.pretrained else None
            backbone = resnet50(weights=weights)
        except Exception:
            try:
                from torchvision.models import resnet50

                backbone = resnet50(pretrained=self.pretrained)
            except Exception:
                if self.prefer_timm:
                    try:
                        import timm

                        self.backbone = timm.create_model(
                            "resnet50",
                            pretrained=self.pretrained,
                            features_only=True,
                            out_indices=(1, 2, 3, 4),
                        )
                        self._backend = "timm:resnet50"
                        self.feature_dims = [256, 512, 1024, 2048]
                        return
                    except Exception as exc:
                        raise ImportError(
                            "Failed to build the image backbone. Install `torchvision` or `timm`."
                        ) from exc
                raise

        self._backend = "torchvision:resnet50"
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.feature_dims = [256, 512, 1024, 2048]

    def _build_vgg16(self) -> None:
        try:
            from torchvision.models import VGG16_Weights, vgg16

            weights = VGG16_Weights.IMAGENET1K_V1 if self.pretrained else None
            backbone = vgg16(weights=weights)
        except Exception:
            from torchvision.models import vgg16

            backbone = vgg16(pretrained=self.pretrained)

        features = backbone.features
        self.layer1 = nn.Sequential(*features[:10])   # 128, 56, 56
        self.layer2 = nn.Sequential(*features[10:17])  # 256, 28, 28
        self.layer3 = nn.Sequential(*features[17:24])  # 512, 14, 14
        self.layer4 = nn.Sequential(*features[24:31])  # 512, 7, 7
        self._backend = "torchvision:vgg16"
        self.feature_dims = [128, 256, 512, 512]

    def _build_mobilenet_v2(self) -> None:
        try:
            from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

            weights = MobileNet_V2_Weights.IMAGENET1K_V2 if self.pretrained else None
            backbone = mobilenet_v2(weights=weights)
        except Exception:
            from torchvision.models import mobilenet_v2

            backbone = mobilenet_v2(pretrained=self.pretrained)

        features = backbone.features
        self.layer1 = nn.Sequential(*features[:4])      # 24, 56, 56
        self.layer2 = nn.Sequential(*features[4:7])     # 32, 28, 28
        self.layer3 = nn.Sequential(*features[7:14])    # 96, 14, 14
        self.layer4 = nn.Sequential(*features[14:])     # 1280, 7, 7
        self._backend = "torchvision:mobilenet_v2"
        self.feature_dims = [24, 32, 96, 1280]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._backend == "timm:resnet50":
            feats = self.backbone(x)
            return feats[0], feats[1], feats[2], feats[3]

        if self.backbone_name == "resnet50":
            x = self.stem(x)
            f1 = self.layer1(x)
            f2 = self.layer2(f1)
            f3 = self.layer3(f2)
            f4 = self.layer4(f3)
            return f1, f2, f3, f4
        if self.backbone_name == "vgg16":
            f1 = self.layer1(x)
            f2 = self.layer2(f1)
            f3 = self.layer3(f2)
            f4 = self.layer4(f3)
            return f1, f2, f3, f4

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return f1, f2, f3, f4

    def freeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    def unfreeze_stage(self, stage_names: Sequence[str]) -> None:
        if self._backend == "timm:resnet50":
            for name, param in self.backbone.named_parameters():
                if any(stage in name for stage in stage_names):
                    param.requires_grad = True
        else:
            for stage in stage_names:
                if hasattr(self, stage):
                    for param in getattr(self, stage).parameters():
                        param.requires_grad = True


class MetadataEncoder(nn.Module):
    """Encode metadata into one token.

    Input:
        meta: [B, meta_in_dim]
    Output:
        q: [B, 256]
    """

    def __init__(self, meta_in_dim: int, embed_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(meta_in_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, meta: torch.Tensor) -> torch.Tensor:
        return self.net(meta)


class HLFDBWrapper(nn.Module):
    """Return explicit high-frequency and low-frequency feature maps.

    Input:
        x: [B, C, H, W]
    Output:
        high: [B, C, H, W]
        low: [B, C, H, W]
    """

    def __init__(self, channels: int, wave: str = "haar"):
        super().__init__()
        self.channels = channels
        self.wave = wave
        self.verbose = False
        self._backend_logged = False
        self._fallback_logged = False
        self.high_reduce = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False)
        self.high_bn = nn.BatchNorm2d(channels)
        self.high_fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.low_proj = _conv_bn_relu(channels, channels, 1)
        self.use_dwt = DWT_2D is not None
        self.dwt = DWT_2D(wave) if self.use_dwt else None

    def set_verbose(self, verbose: bool = True) -> None:
        self.verbose = verbose

    def _log_backend_once(self) -> None:
        if self.verbose and not self._backend_logged:
            backend = "DWT_2D" if self.use_dwt else "fallback"
            print(f"HLFDBWrapper backend: {backend} (wave={self.wave})")
            if HLFDB_LOAD_ERROR:
                print(f"HLFDB.py load error: {HLFDB_LOAD_ERROR}")
            self._backend_logged = True

    def _log_fallback_once(self, reason: str) -> None:
        if self.verbose and not self._fallback_logged:
            print(f"HLFDBWrapper fallback activated: {reason}")
            self._fallback_logged = True

    def _fallback(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        low = F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
        low = F.interpolate(low, size=x.shape[-2:], mode="bilinear", align_corners=False)
        low = self.low_proj(low)
        high_signed = x - low
        high_mag = torch.abs(high_signed)
        high = self.high_fuse(torch.cat([high_signed, high_mag], dim=1))
        return high, low

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._log_backend_once()
        if not self.use_dwt:
            self._log_fallback_once("DWT_2D unavailable")
            return self._fallback(x)

        orig_h, orig_w = x.shape[-2:]
        pad_h = orig_h % 2
        pad_w = orig_w % 2
        x_padded = x
        if pad_h or pad_w:
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        try:
            dwt_out = self.dwt(x_padded)
            c = x_padded.shape[1]
            ll, lh, hl, hh = dwt_out.split(c, dim=1)
            low = self.low_proj(ll)
            high_signed = self.high_bn(self.high_reduce(torch.cat([lh, hl, hh], dim=1)))
            high_mag = torch.abs(high_signed)
            high = self.high_fuse(torch.cat([high_signed, high_mag], dim=1))
            low = F.interpolate(low, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            high = F.interpolate(high, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            return high, low
        except Exception as exc:
            self._log_fallback_once(f"{type(exc).__name__}: {exc}")
            return self._fallback(x)


class BFAMWrapper(nn.Module):
    """Texture enhancer for the high-frequency branch.

    Input:
        high: [B, C, H, W]
        low:  [B, C, H, W]
    Output:
        high_enhanced: [B, C, H, W]
    """

    def __init__(self, channels: int):
        super().__init__()
        self.use_full_bfam = B2C_BFAM is not None
        self.bfam = B2C_BFAM(inp=channels * 2, out=channels) if self.use_full_bfam else None
        self.diff = None if self.use_full_bfam else (diff_moudel(channels) if diff_moudel is not None else None)
        self.cbm = None if self.use_full_bfam else (CBM(channels) if CBM is not None else None)
        self.using_full_bfam = self.bfam is not None
        self.bfam_scale = nn.Parameter(torch.tensor(0.2))
        self.refine = _conv_bn_relu(channels, channels, 3)

    def forward(self, high: torch.Tensor, low: torch.Tensor | None = None) -> torch.Tensor:
        if low is None:
            low = F.avg_pool2d(high, kernel_size=3, stride=1, padding=1)

        if self.bfam is not None:
            bfam_out = self.bfam(high, low)
            if bfam_out.shape[-2:] != high.shape[-2:]:
                bfam_out = F.interpolate(bfam_out, size=high.shape[-2:], mode="bilinear", align_corners=False)
            out = high + self.bfam_scale * bfam_out
            return self.refine(out)

        detail = self.diff(high) if self.diff is not None else high
        if self.cbm is not None:
            contrast = self.cbm(high, low)
            if contrast.shape[-2:] != high.shape[-2:]:
                contrast = F.interpolate(contrast, size=high.shape[-2:], mode="bilinear", align_corners=False)
            out = high + 0.5 * detail + 0.25 * contrast
        else:
            out = high + 0.5 * detail
        return self.refine(out)


class CrossModalAgentAttention(nn.Module):
    """Metadata-guided QKV cross-modal attention.

    Input:
        x: [B, C, H, W]
        q_branch: [B, C] or [B, meta_dim]
    Output:
        x_out: [B, C, H, W]
    """

    def __init__(
        self,
        channels: int = 256,
        q_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        min_pool_size: int = 7,
        max_pool_size: int = 14,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads}).")
        self.channels = channels
        self.q_dim = q_dim
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size

        self.q_proj = nn.Linear(q_dim, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.spatial_scale = nn.Parameter(torch.tensor(0.5))
        self.channel_scale = nn.Parameter(torch.tensor(0.5))
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def _pool_size(self, h: int, w: int) -> Tuple[int, int]:
        target_h = min(self.max_pool_size, h)
        target_w = min(self.max_pool_size, w)
        target_h = max(self.min_pool_size if h >= self.min_pool_size else h, min(target_h, h))
        target_w = max(self.min_pool_size if w >= self.min_pool_size else w, min(target_w, w))
        return target_h, target_w

    def forward(self, x: torch.Tensor, q_branch: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        pool_h, pool_w = self._pool_size(h, w)
        pooled = F.adaptive_avg_pool2d(x, (pool_h, pool_w))
        tokens = pooled.flatten(2).transpose(1, 2)  # [B, N, C]
        n = tokens.shape[1]

        q = self.q_proj(q_branch).view(b, self.num_heads, 1, self.head_dim)
        k = self.k_proj(tokens).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(tokens).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, heads, 1, N]
        attn = torch.softmax(attn, dim=-1)
        context = torch.matmul(attn, v).reshape(b, c)

        spatial_attn = attn.mean(dim=1).view(b, 1, pool_h, pool_w)
        spatial_attn = (spatial_attn - spatial_attn.mean(dim=(2, 3), keepdim=True)) / (
            spatial_attn.std(dim=(2, 3), keepdim=True) + 1e-6
        )
        spatial_attn = F.interpolate(spatial_attn, size=(h, w), mode="bilinear", align_corners=False)
        spatial_gate = self.spatial_proj(spatial_attn)
        channel_gate = self.channel_mlp(context).view(b, c, 1, 1)

        out = x + self.spatial_scale * x * spatial_gate + self.channel_scale * x * channel_gate
        out = self.out_proj(out)
        return out


class SingleScaleTFPFM(nn.Module):
    """Single-scale Texture-Frequency Progressive Fusion Module."""

    def __init__(
        self,
        channels: int = 256,
        meta_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_sade: bool = True,
        use_metadata: bool = True,
        use_mgca: bool = True,
        use_dsdm: bool = True,
    ):
        super().__init__()
        if use_mgca and not use_metadata:
            raise ValueError("MGCA requires metadata; set use_mgca=False when use_metadata=False.")
        self.freq = HLFDBWrapper(channels) if use_dsdm else None
        self.use_sade = use_sade
        self.use_metadata = use_metadata
        self.use_mgca = use_mgca
        self.use_dsdm = use_dsdm
        self.bfam = BFAMWrapper(channels) if use_sade else None
        self.q_high = nn.Linear(meta_dim, channels) if use_mgca else None
        self.q_low = nn.Linear(meta_dim, channels) if use_mgca else None
        self.q_proj = nn.Linear(meta_dim, channels) if use_metadata else None
        self.high_attn = (
            CrossModalAgentAttention(channels=channels, q_dim=channels, num_heads=num_heads, dropout=dropout)
            if use_mgca else None
        )
        self.low_attn = (
            CrossModalAgentAttention(channels=channels, q_dim=channels, num_heads=num_heads, dropout=dropout)
            if use_mgca else None
        )
        fuse_in_channels = channels * (3 if use_metadata else 2)
        self.fuse1 = _conv_bn_relu(fuse_in_channels, channels, 1)
        self.fuse2 = _conv_bn_relu(channels, channels, 3)
        self.mscb = MSCBLayer(channels, channels, n=1, stride=1, kernel_sizes=[1, 3, 5], expansion_factor=2)
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def set_verbose(self, verbose: bool = True) -> None:
        if self.freq is not None:
            self.freq.set_verbose(verbose)

    def forward(self, x: torch.Tensor, q: torch.Tensor | None = None) -> torch.Tensor:
        b, _, h, w = x.shape
        if self.use_dsdm:
            high, low = self.freq(x)
        else:
            high, low = x, x
        if self.use_sade and self.bfam is not None:
            high = self.bfam(high, low)
        if self.use_mgca:
            if q is None:
                raise ValueError("Metadata embedding is required when MGCA is enabled.")
            high = self.high_attn(high, self.q_high(q))
            low = self.low_attn(low, self.q_low(q))

        fusion_inputs = [high, low]
        if self.use_metadata:
            if q is None:
                raise ValueError("Metadata embedding is required when metadata fusion is enabled.")
            q_map = self.q_proj(q).view(b, -1, 1, 1).expand(-1, -1, h, w)
            fusion_inputs.append(q_map)

        fused = torch.cat(fusion_inputs, dim=1)
        fused = self.fuse1(fused)
        fused = self.fuse2(fused)
        fused = self.mscb(fused)
        if fused.shape[-2:] != x.shape[-2:]:
            fused = F.interpolate(fused, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x + self.res_scale * fused


class HFPA_Net(nn.Module):
    """Hierarchical Frequency-aware Progressive Attention Fusion Network."""

    def __init__(
        self,
        num_classes: int = 6,
        meta_in_dim: int = 81,
        fusion_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        pretrained_backbone: bool = True,
        backbone_name: str = "resnet50",
        use_all_afpn_outputs: bool = False,
        use_sade: bool = True,
        use_metadata: bool = True,
        use_mgca: bool = True,
        use_dsdm: bool = True,
        use_dsmaf: bool = True,
        use_afpn: bool = True,
        use_cluster_aux_head: bool = False,
        verbose: bool = False,
    ):
        super().__init__()
        if not use_dsmaf:
            use_sade = False
            use_mgca = False
            use_dsdm = False
        self.num_classes = num_classes
        self.meta_in_dim = meta_in_dim
        self.fusion_dim = fusion_dim
        self.use_all_afpn_outputs = use_all_afpn_outputs
        self.use_sade = use_sade
        self.use_metadata = use_metadata
        self.use_mgca = use_mgca
        self.use_dsdm = use_dsdm
        self.use_dsmaf = use_dsmaf
        self.use_afpn = use_afpn
        self.use_late_metadata_fusion = (not use_dsmaf) and use_metadata
        self.use_cluster_aux_head = use_cluster_aux_head
        self.verbose = verbose

        if use_mgca and not use_metadata:
            raise ValueError("MGCA requires metadata; set use_mgca=False when use_metadata=False.")

        self.image_backbone = ImageBackbone(backbone_name=backbone_name, pretrained=pretrained_backbone)
        self.meta_encoder = (
            MetadataEncoder(meta_in_dim=meta_in_dim, embed_dim=fusion_dim, dropout=0.2)
            if use_metadata else None
        )
        f1_dim, f2_dim, f3_dim, f4_dim = self.image_backbone.feature_dims
        self.adapter1 = _conv_bn_relu(f1_dim, fusion_dim, 1)
        self.adapter2 = _conv_bn_relu(f2_dim, fusion_dim, 1)
        self.adapter3 = _conv_bn_relu(f3_dim, fusion_dim, 1)
        self.adapter4 = _conv_bn_relu(f4_dim, fusion_dim, 1)

        if use_dsmaf:
            block_kwargs = dict(
                channels=fusion_dim,
                meta_dim=fusion_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_sade=use_sade,
                use_metadata=use_metadata,
                use_mgca=use_mgca,
                use_dsdm=use_dsdm,
            )
            self.tfpfm1 = SingleScaleTFPFM(**block_kwargs)
            self.tfpfm2 = SingleScaleTFPFM(**block_kwargs)
            self.tfpfm3 = SingleScaleTFPFM(**block_kwargs)
            self.tfpfm4 = SingleScaleTFPFM(**block_kwargs)
        else:
            self.tfpfm1 = None
            self.tfpfm2 = None
            self.tfpfm3 = None
            self.tfpfm4 = None

        self.afpn = (
            AFPN(in_channels=[fusion_dim, fusion_dim, fusion_dim, fusion_dim], out_channels=fusion_dim)
            if use_afpn else None
        )
        self.da_block = DA_Block(fusion_dim) if DA_Block is not None else _conv_bn_relu(fusion_dim, fusion_dim, 3)
        self.da_res_scale = nn.Parameter(torch.tensor(0.5))

        image_feature_dim = fusion_dim * 4 if use_all_afpn_outputs else fusion_dim
        classifier_in_dim = image_feature_dim + (fusion_dim if self.use_late_metadata_fusion else 0)
        self.classifier = nn.Linear(classifier_in_dim, num_classes)
        self.cluster_aux_head = nn.Linear(classifier_in_dim, 3) if use_cluster_aux_head else None
        self._bfam_backend_logged = False

    def set_verbose(self, verbose: bool = True) -> None:
        self.verbose = verbose
        if self.use_dsmaf:
            self.tfpfm1.set_verbose(verbose)
            self.tfpfm2.set_verbose(verbose)
            self.tfpfm3.set_verbose(verbose)
            self.tfpfm4.set_verbose(verbose)

    def freeze_backbone(self, mode: str = "all") -> None:
        self.image_backbone.unfreeze_all()
        if mode == "none":
            return
        if mode == "all":
            self.image_backbone.freeze_all()
            return
        self.image_backbone.freeze_all()
        if mode == "layer3_layer4":
            self.image_backbone.unfreeze_stage(["layer3", "layer4"])
        elif mode == "layer4":
            self.image_backbone.unfreeze_stage(["layer4"])
        else:
            raise ValueError(f"Unsupported freeze mode: {mode}")

    def _maybe_log(self, name: str, tensor: torch.Tensor) -> None:
        if self.verbose:
            print(f"{name}: {tuple(tensor.shape)}")

    def forward(self, img: torch.Tensor, meta: torch.Tensor | None):
        if self.use_metadata:
            if meta is None or meta.ndim != 2:
                shape = None if meta is None else tuple(meta.shape)
                raise ValueError(f"Expected metadata tensor with shape [B, meta_in_dim], got {shape}")
            if meta.shape[1] != self.meta_in_dim:
                raise ValueError(
                    f"Metadata dimension mismatch: expected {self.meta_in_dim}, got {meta.shape[1]}"
                )

        f1, f2, f3, f4 = self.image_backbone(img)
        self._maybe_log("F1", f1)
        self._maybe_log("F2", f2)
        self._maybe_log("F3", f3)
        self._maybe_log("F4", f4)
        if self.verbose and self.use_dsmaf and not self._bfam_backend_logged:
            if self.use_sade and self.tfpfm1.bfam is not None:
                print(f"SADE/BFAMWrapper enabled; using full B2C_BFAM: {self.tfpfm1.bfam.using_full_bfam}")
            else:
                print("SADE/BFAMWrapper disabled")
            self._bfam_backend_logged = True

        q = self.meta_encoder(meta) if self.use_metadata else None
        f1 = self.adapter1(f1)
        f2 = self.adapter2(f2)
        f3 = self.adapter3(f3)
        f4 = self.adapter4(f4)

        if self.use_dsmaf:
            g1 = self.tfpfm1(f1, q)
            g2 = self.tfpfm2(f2, q)
            g3 = self.tfpfm3(f3, q)
            g4 = self.tfpfm4(f4, q)
        else:
            g1, g2, g3, g4 = f1, f2, f3, f4
        self._maybe_log("G1", g1)
        self._maybe_log("G2", g2)
        self._maybe_log("G3", g3)
        self._maybe_log("G4", g4)

        pyramid_outs = self.afpn([g1, g2, g3, g4]) if self.use_afpn else (g1, g2, g3, g4)
        if len(pyramid_outs) != 4:
            raise RuntimeError(f"Expected 4 pyramid outputs, got {len(pyramid_outs)}")
        for idx, feat in enumerate(pyramid_outs):
            self._maybe_log(f"Pyramid_out{idx}", feat)

        p_base = pyramid_outs[3]
        p_refine = self.da_block(p_base)
        if p_refine.shape[-2:] != p_base.shape[-2:]:
            p_refine = F.interpolate(p_refine, size=p_base.shape[-2:], mode="bilinear", align_corners=False)
        p = p_base + self.da_res_scale * p_refine
        self._maybe_log("DA_out", p)

        if self.use_all_afpn_outputs:
            pooled = [
                F.adaptive_avg_pool2d(feat if idx != 3 else p, 1).flatten(1)
                for idx, feat in enumerate(pyramid_outs)
            ]
            fused_feat = torch.cat(pooled, dim=1)
        else:
            fused_feat = F.adaptive_avg_pool2d(p, 1).flatten(1)

        if self.use_late_metadata_fusion:
            if q is None:
                raise RuntimeError("Late metadata fusion requires a metadata embedding.")
            fused_feat = torch.cat([fused_feat, q], dim=1)

        logits = self.classifier(fused_feat)
        self._maybe_log("logits", logits)
        if self.cluster_aux_head is None:
            return logits

        aux_cluster_logits = self.cluster_aux_head(fused_feat)
        self._maybe_log("aux_cluster_logits", aux_cluster_logits)
        return {"logits": logits, "aux_cluster_logits": aux_cluster_logits}


if __name__ == "__main__":
    meta_in_dim = 81
    model = HFPA_Net(
        num_classes=6,
        meta_in_dim=meta_in_dim,
        pretrained_backbone=False,
        use_all_afpn_outputs=True,
        verbose=True,
    )
    img = torch.randn(2, 3, 224, 224)
    meta = torch.randn(2, meta_in_dim)
    logits = model(img, meta)
    assert logits.shape == (2, 6), f"Unexpected output shape: {tuple(logits.shape)}"
    print("HFPA_Net forward test passed:", logits.shape)
