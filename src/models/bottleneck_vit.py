import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .base import BaseModel
except ImportError:
    from base import BaseModel


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
        self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
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
                )
                for _ in range(depth)
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
    """Concatenation-based Pyramid Feature Extractor for better SE reweighting."""

    def __init__(self, input_channels=3):
        super().__init__()
        # Bottom-up pathway
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3
            ),  # 64x64 -> 32x32
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        # Bottom-up pathway
        c1 = self.layer1(x)  # 32x32, 64ch
        c2 = self.layer2(c1)  # 16x16, 128ch
        c3 = self.layer3(c2)  # 8x8, 256ch

        # Upsample all to 32x32 and concatenate
        c2_up = F.interpolate(
            c2, size=c1.shape[2:], mode="bilinear", align_corners=False
        )
        c3_up = F.interpolate(
            c3, size=c1.shape[2:], mode="bilinear", align_corners=False
        )

        # Concatenate: 64 + 128 + 256 = 448 channels
        fused = torch.cat([c1, c2_up, c3_up], dim=1)  # 32x32, 448ch
        return fused


class BottleneckViT(BaseModel):
    """Vision Transformer with Bottleneck for Chinese Character Recognition.

    Structure: Conv feature extraction -> Conv bottleneck -> ViT blocks -> 2x FC classification
    """

    def __init__(
        self,
        img_size=64,
        patch_size=8,
        input_channels=3,
        num_classes=631,
        embed_dim=256,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.2,
        **kwargs,
    ):
        super().__init__(
            num_classes=num_classes, input_channels=input_channels, **kwargs
        )

        self.embed_dim = embed_dim

        # Stage 1: Pyramid Feature Extractor (outputs 448 channels)
        self.pyramid_extractor = PyramidFeatureExtractor(input_channels=input_channels)

        # Stage 2: SE Block on concatenated features + Channel Reduction
        # 448 channels: SE can learn to reweight each scale independently
        self.se_block = SEBlock(448, reduction=32)  # Larger reduction for more channels
        self.channel_reduction = nn.Sequential(
            nn.Conv2d(448, embed_dim, kernel_size=1),  # 448 -> embed_dim
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )
        self.conv_bottleneck = nn.Sequential(
            nn.Conv2d(
                embed_dim, embed_dim, kernel_size=3, stride=2, padding=1
            ),  # 32x32 -> 16x16 (256 patches for ViT)
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )

        # Stage 3: Vision Transformer
        # After Stage 1 (stride=2): 64x64 -> 32x32
        # After Stage 2 (stride=2): 32x32 -> 16x16
        # Total patches: 16x16 = 256 (4x more than before for better detail preservation)
        num_patches = 256
        self.vit = VisionTransformer(
            embed_dim=embed_dim,
            num_patches=num_patches,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
        )

        # Stage 5: Classification Head (2x FC with more capacity)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim, num_classes),
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
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Stage 1: FPN Feature Extraction
        x = self.pyramid_extractor(x)  # 32x32, embed_dim

        # Stage 2: SE Block + Channel Reduction + Conv Bottleneck
        x = self.se_block(x)  # SE reweights all 448 channels
        x = self.channel_reduction(x)  # 448 -> embed_dim
        x = self.conv_bottleneck(x)

        # Stage 3: Reshape to patches for ViT (B, C, H, W) -> (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)

        # Stage 4: ViT Processing
        x = self.vit(x)

        # Stage 5: Classification
        return self.head(x[:, 0])


if __name__ == "__main__":
    from torchinfo import summary

    model = BottleneckViT()
    summary(model, input_size=(1, 3, 64, 64))
