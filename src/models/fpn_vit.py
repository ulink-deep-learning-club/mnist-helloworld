import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

try:
    from .base import BaseModel
    from ..training.metrics import MetricsTracker
except ImportError:
    from base import BaseModel
    from training.metrics import MetricsTracker


class PatchEmbed(nn.Module):
    """将 2D 图片展平为 Patch Embeddings"""

    def __init__(self, img_size=64, patch_size=8, in_chans=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size**2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # kernel function：elu+1（确保非负）
        self.kernel_fn = nn.ELU()

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, head_dim]

        q = self.kernel_fn(q) + 1
        k = self.kernel_fn(k) + 1

        # (Q @ (K^T @ V)) / scaling
        q = q * self.scale
        # K^T @ V  [B, heads, head_dim, head_dim]
        kv = k.transpose(-2, -1) @ v
        # Q @ KV  [B, heads, N, head_dim]
        x = q @ kv

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FocusedLinearAttention(nn.Module):
    """
    Focused Linear Attention (ICCV 2023)
    核心改进：加入 focusing factor 让注意力分布更尖锐
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        focusing_factor=3,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.focusing_factor = focusing_factor

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # learnable focusing param
        self.dwc = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # (B, heads, N, d)

        # 1. ReLU kernel mapping (ensure non-negative)
        q = F.relu(q) + 1e-6
        k = F.relu(k) + 1e-6

        # 2. L2 norm + focusing factor (make distribution sharper)
        q = q / q.sum(dim=-1, keepdim=True)  # normalize along feature dim
        k = k / k.sum(dim=-1, keepdim=True)
        q = q**self.focusing_factor  # power operation to make distribution sharper
        k = k**self.focusing_factor
        q = q / q.sum(dim=-1, keepdim=True)  # re-normalize
        k = k / k.sum(dim=-1, keepdim=True)

        # 3. linear attention: first calc K^T @ V, complexity O(Nd^2)
        kv = torch.einsum("bhnd,bhne->bhde", k, v)  # (B, h, d, d)
        z = 1.0 / (
            torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2)) + 1e-6
        )  # normalization factor
        out = torch.einsum("bhnd,bhde,bhn->bhne", q, kv, z)  # (B, h, N, d)

        # 4. Local feature enhancement (compensate for lost local info in linear attention)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = out + self.dwc(out.transpose(1, 2)).transpose(1, 2)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        linear_attention=False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = (
            Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
            if not linear_attention
            else FocusedLinearAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    """Standard Conv Block (Conv + BatchNorm + Activation)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        groups=1,
        act=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """YOLOv5 Bottleneck block with optional shortcut."""

    def __init__(
        self, in_channels, out_channels, shortcut=True, groups=1, expansion=0.5
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = ConvBlock(in_channels, hidden_channels, 1, 1)
        self.cv2 = ConvBlock(hidden_channels, out_channels, 3, 1, 1, groups=groups)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3Module(nn.Module):
    """YOLOv5 C3 Module (Cross Stage Partial Network).

    Args:
        in_channels: Input channel count
        out_channels: Output channel count
        num_bottlenecks: Number of bottleneck blocks
        shortcut: Whether to use shortcut connections in bottlenecks
        groups: Number of groups for grouped convolutions
        expansion: Channel expansion factor for bottleneck hidden layers
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_bottlenecks=3,
        shortcut=True,
        groups=1,
        expansion=0.5,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = ConvBlock(in_channels, hidden_channels, 1, 1)
        self.cv2 = ConvBlock(in_channels, hidden_channels, 1, 1)
        self.cv3 = ConvBlock(2 * hidden_channels, out_channels, 1, 1)
        self.m = nn.Sequential(
            *[
                Bottleneck(
                    hidden_channels, hidden_channels, shortcut, groups, expansion
                )
                for _ in range(num_bottlenecks)
            ]
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class VisionTransformer(nn.Module):
    """Vision Transformer module for processing patch embeddings."""

    def __init__(
        self,
        embed_dim=256,
        num_patches=256,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.2,
        linear_attention=False,
        linear_layer_limit=4,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    linear_attention=linear_attention
                    if i < linear_layer_limit
                    else False,
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x


class PyramidFeatureExtractor(nn.Module):
    """Concatenation-based Pyramid Feature Extractor with YOLOv5 C3 modules."""

    def __init__(
        self, input_channels=3, lateral_channels_list=[64, 128, 256], out_dim=256
    ):
        super().__init__()
        # Stage 1: Conv + C3 (downsample from 64x64 to 32x32)
        self.stem = ConvBlock(
            input_channels, lateral_channels_list[0], 3, 2, 1
        )  # 64x64 -> 32x32
        self.layer1 = C3Module(
            lateral_channels_list[0],
            lateral_channels_list[0],
            num_bottlenecks=3,
            shortcut=True,
        )

        # Stage 2: Conv + C3 (downsample from 32x32 to 16x16)
        self.down2 = ConvBlock(
            lateral_channels_list[0], lateral_channels_list[1], 3, 2, 1
        )  # 32x32 -> 16x16
        self.layer2 = C3Module(
            lateral_channels_list[1],
            lateral_channels_list[1],
            num_bottlenecks=3,
            shortcut=True,
        )

        # Stage 3: Conv + C3 (downsample from 16x16 to 8x8)
        self.down3 = ConvBlock(
            lateral_channels_list[1], lateral_channels_list[2], 3, 2, 1
        )  # 16x16 -> 8x8
        self.layer3 = C3Module(
            lateral_channels_list[2],
            lateral_channels_list[2],
            num_bottlenecks=3,
            shortcut=True,
        )

        self.lateral1 = nn.Sequential(
            nn.Conv2d(lateral_channels_list[0], out_dim, 1),
            nn.AvgPool2d(2),
        )
        self.lateral2 = nn.Conv2d(lateral_channels_list[1], out_dim, 1)
        self.lateral3 = nn.Sequential(
            nn.Conv2d(lateral_channels_list[2], out_dim, 1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

        # Scale-specific SE (optional shared weights)
        self.se1 = SEBlock(out_dim)
        self.se2 = SEBlock(out_dim)
        self.se3 = SEBlock(out_dim)

        # Fusion conv: reduce channels after concatenation
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_dim * 3, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        # Bottom-up pathway with C3 modules
        x = self.stem(x)  # 32x32, 64ch
        c1 = self.layer1(x)  # 32x32, 64ch
        x = self.down2(c1)  # 16x16, 128ch
        c2 = self.layer2(x)  # 16x16, 128ch
        x = self.down3(c2)  # 8x8, 256ch
        c3 = self.layer3(x)  # 8x8, 256ch

        # 1. Unify channels
        p1 = self.lateral1(c1)  # 32x32, out_dim
        p2 = self.lateral2(c2)  # 16x16, out_dim
        p3 = self.lateral3(c3)  # 8x8, out_dim

        # 2. Apply SE independently
        p1 = self.se1(p1)
        p2 = self.se2(p2)
        p3 = self.se3(p3)

        # 4. Concatenate and fuse
        fused = torch.cat([p1, p2, p3], dim=1)  # (B, out_dim*3, 16, 16)
        out = self.fusion_conv(fused)  # (B, out_dim, 16, 16)
        return out


class FeaturePyramidViT(BaseModel):
    """Vision Transformer with Bottleneck for Chinese Character Recognition.

    Structure: Conv feature extraction -> Conv bottleneck -> ViT blocks -> 2x FC classification
    """

    @property
    def model_type(self) -> str:
        return "classification"

    @classmethod
    def get_criterion(cls, **kwargs) -> nn.Module:
        """Return CrossEntropyLoss for classification."""
        return nn.CrossEntropyLoss()

    @classmethod
    def get_metrics_tracker(cls, **kwargs) -> Any:
        """Return standard metrics tracker for classification."""
        return MetricsTracker()

    def __init__(
        self,
        img_size=64,
        preprocess_channels=32,
        fpn_out_channels=128,
        embed_dim=128,
        patch_size=16,
        input_channels=3,
        num_classes=631,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.2,
        **kwargs,
    ):
        super().__init__(
            num_classes=num_classes, input_channels=input_channels, **kwargs
        )

        self.embed_dim = embed_dim

        self.image_preprocess = nn.Sequential(
            nn.Conv2d(input_channels, preprocess_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(preprocess_channels),
            nn.SiLU(inplace=True),
        )

        # Stage 1: Pyramid Feature Extractor & Channel Reduction
        self.pyramid_extractor = PyramidFeatureExtractor(
            input_channels=preprocess_channels,
            lateral_channels_list=[64, 128, 256],
            out_dim=fpn_out_channels,
        )

        # Stage 2: 256 patches for ViT
        stride = (img_size // 4) // patch_size
        num_patches = ((img_size // 4) // stride) ** 2
        self.conv_bottleneck = nn.Sequential(
            nn.Conv2d(
                fpn_out_channels,
                self.embed_dim,
                kernel_size=3,
                padding=1,
                stride=stride,
            ),
            nn.BatchNorm2d(self.embed_dim),
            nn.SiLU(inplace=True),
        )

        # Stage 3: Vision Transformer
        # Total patches: 16x16 = 256 (4x more than before for better detail preservation)
        self.vit = VisionTransformer(
            embed_dim=self.embed_dim,
            num_patches=num_patches,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            linear_attention=True,
            linear_layer_limit=4,
        )

        # Stage 5: Classification Head (2x FC with more capacity)
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(self.embed_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        self.apply(self._init_weights_layer)

    def _init_weights_layer(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.image_preprocess(x)

        # Stage 1: FPN Feature Extraction
        x = self.pyramid_extractor(x)

        # Stage 2: Conv Bottleneck
        x = self.conv_bottleneck(x)

        # Stage 3: Reshape to patches for ViT (B, C, H, W) -> (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)

        # Stage 4: ViT Processing
        x = self.vit(x)

        # Stage 5: Classification
        return self.head(x[:, 0])


if __name__ == "__main__":
    from torchinfo import summary

    model = FeaturePyramidViT()
    summary(model, input_size=(1, 3, 64, 64))
