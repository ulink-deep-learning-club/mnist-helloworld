"""简化版 LeNet 激活函数对比系列 (11 个 profile).

设计目标: 模型容量故意做小 (约 5.4k 参数, 约为经典 LeNet 的 1/10),
无 BN、无 Dropout, 让模型在 MNIST 上"吃力" (冲不高分),
从而放大不同激活函数带来的差异.

结构 (以 28x28 输入为例):
    Conv(1->4, k5) -> Act -> AvgPool(2)     # 28x28 -> 24x24 -> 12x12
    Conv(4->8, k5) -> Act -> AvgPool(2)     # 12x12 ->  8x8  ->  4x4
    Flatten(8*4*4=128) -> Linear(128->32) -> Act -> Linear(32->10)

11 个 profile 除了激活函数外结构、初始化完全一致:
    lenet_act_sigmoid      : nn.Sigmoid()
    lenet_act_tanh         : nn.Tanh()
    lenet_act_arctan       : atan(x)
    lenet_act_relu         : nn.ReLU()
    lenet_act_lrelu        : nn.LeakyReLU(0.01)
    lenet_act_silu         : nn.SiLU() (即 x*sigmoid(x), 恰为 softlrelu 取 beta=0 时的退化形态)
    lenet_act_calu         : CaLU (即 x*CauchyCDF(x), x·CDF 家族第三块拼图: SiLU/GELU/CaLU)
    lenet_act_leakycau     : LeakyCaLU (即 (1-beta)*x*CauchyCDF(x) + beta*x, arctan 门控的 soft leaky)
    lenet_act_softlrelu    : (1-beta)*x*sigmoid(x) + beta*x, beta=0.01
    lenet_act_roclu        : RoCLU (即 1.444*x*ln(1.5 + atan(x)/pi), 文献基线, CCPE 2026 e70944)
    lenet_act_gelu         : nn.GELU() (精确 erf 版, x·GaussianCDF, 三位一体之二)
                             (负半轴渐近斜率为 beta, 与 LeakyReLU(0.01) 对齐;
                              beta=0 退化为 SiLU/Swish, beta=1 退化为恒等映射)

初始化统一用 Xavier 均匀分布 + 零偏置, 对所有激活函数保持中立.
"""

import math

import torch
import torch.nn as nn
from typing import Any

from .base import BaseModel
from ..training.metrics import MetricsTracker

SOFT_LEAKY_BETA = 0.01
ROCLU_K = 1.444


class Arctan(nn.Module):
    """反正切激活: f(x) = atan(x), 值域 (-pi/2, pi/2)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.atan(x)


class SoftLeakyReLU(nn.Module):
    """Soft Leaky ReLU: f(x) = (1-beta) * x * sigmoid(x) + beta * x.

    x -> +inf 时趋近于 y = x; x -> -inf 时趋近于 y = beta * x,
    是 LeakyReLU 的光滑 (处处可导) 版本.
    """

    def __init__(self, beta: float = SOFT_LEAKY_BETA):
        super().__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.beta) * x * torch.sigmoid(x) + self.beta * x

    def extra_repr(self) -> str:
        return f"beta={self.beta}"


def _cauchy_cdf(x: torch.Tensor) -> torch.Tensor:
    """标准 Cauchy 分布的 CDF: C(x) = 1/2 + arctan(x)/pi, 值域 (0, 1)."""
    return (torch.atan(x) / (math.pi / 2) + 1) / 2


class CaLU(nn.Module):
    """Cauchy Linear Unit: f(x) = x * C(x).

    x·CDF(x) 家族的第三块拼图 (Logistic CDF -> SiLU/Swish,
    Gaussian CDF -> GELU, Cauchy CDF -> CaLU). 重尾门控, 严格单调.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * _cauchy_cdf(x)


class LeakyCaLU(nn.Module):
    """Leaky CaLU: f(x) = (1-beta) * x * C(x) + beta * x.

    CaLU 的 leak 化版本, 与 SoftLeakyReLU 同模板 (仅门控不同:
    Cauchy CDF vs Sigmoid), beta 与 LeakyReLU 负斜率对齐.
    """

    def __init__(self, beta: float = SOFT_LEAKY_BETA):
        super().__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.beta) * x * _cauchy_cdf(x) + self.beta * x

    def extra_repr(self) -> str:
        return f"beta={self.beta}"


def _roclu_gate(x: torch.Tensor, k: float = ROCLU_K) -> torch.Tensor:
    """RoCLU 的门控: g(x) = k * ln(1.5 + atan(x)/pi), 值域 (0, 1), 上界归一≈1."""
    return k * torch.log(1.5 + torch.atan(x) / math.pi)


class RoCLU(nn.Module):
    """Robust Cauchy Linear Unit: f(x) = k * x * ln(1.5 + atan(x)/pi), k=1.444.

    文献基线 (Kucukali et al., Concurr. Comput. Pract. Exp. 2026, 38:e70944).
    与 CaLU 同属 x·g(atan(x)) 重尾门控家族, 门控更重、下界更深 (约 -0.46).
    注意 log 真数恒在 (1, 2) 内, 无数值稳定性问题.
    """

    def __init__(self, k: float = ROCLU_K):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _roclu_gate(x, self.k) * x

    def extra_repr(self) -> str:
        return f"k={self.k}"


def build_activation(name: str) -> nn.Module:
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    if name == "arctan":
        return Arctan()
    if name == "relu":
        return nn.ReLU()
    if name == "lrelu":
        return nn.LeakyReLU(negative_slope=0.01)
    if name == "silu":
        return nn.SiLU()
    if name == "calu":
        return CaLU()
    if name == "leakycau":
        return LeakyCaLU(beta=SOFT_LEAKY_BETA)
    if name == "roclu":
        return RoCLU(k=ROCLU_K)
    if name == "gelu":
        return nn.GELU()
    if name == "softlrelu":
        return SoftLeakyReLU(beta=SOFT_LEAKY_BETA)
    raise ValueError(f"Unknown activation: {name}")


class _SimplifiedLeNetAct(BaseModel):
    """简化版 LeNet 基类, 子类只需指定 ACT_NAME."""

    ACT_NAME: str = "relu"

    @property
    def model_type(self) -> str:
        return "classification"

    @classmethod
    def get_criterion(cls, **kwargs) -> nn.Module:
        return nn.CrossEntropyLoss()

    @classmethod
    def get_metrics_tracker(cls, **kwargs) -> Any:
        return MetricsTracker()

    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 1,
        input_size: tuple = (28, 28),
        **kwargs,
    ):
        super().__init__(num_classes, input_channels)
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = tuple(input_size)

        self.conv1 = nn.Conv2d(input_channels, 4, kernel_size=5)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        h, w = self.input_size
        h = (h - 5 + 1) // 2  # conv1 + pool1
        h = (h - 5 + 1) // 2  # conv2 + pool2
        w = (w - 5 + 1) // 2
        w = (w - 5 + 1) // 2
        flattened = 8 * h * w

        self.fc1 = nn.Linear(flattened, 32)
        self.fc2 = nn.Linear(32, num_classes)

        # 对所有激活函数统一使用 Xavier 均匀初始化, 保证公平
        for m in (self.conv1, self.conv2, self.fc1, self.fc2):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _act(self) -> nn.Module:
        return build_activation(self.ACT_NAME)

    def forward(self, x):
        x = self.pool1(self._act()(self.conv1(x)))
        x = self.pool2(self._act()(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self._act()(self.fc1(x))
        return self.fc2(x)


class LeNetActSigmoid(_SimplifiedLeNetAct):
    ACT_NAME = "sigmoid"


class LeNetActTanh(_SimplifiedLeNetAct):
    ACT_NAME = "tanh"


class LeNetActArctan(_SimplifiedLeNetAct):
    ACT_NAME = "arctan"


class LeNetActReLU(_SimplifiedLeNetAct):
    ACT_NAME = "relu"


class LeNetActLeakyReLU(_SimplifiedLeNetAct):
    ACT_NAME = "lrelu"


class LeNetActSiLU(_SimplifiedLeNetAct):
    ACT_NAME = "silu"


class LeNetActCaLU(_SimplifiedLeNetAct):
    ACT_NAME = "calu"


class LeNetActLeakyCaLU(_SimplifiedLeNetAct):
    ACT_NAME = "leakycau"


class LeNetActRoCLU(_SimplifiedLeNetAct):
    ACT_NAME = "roclu"


class LeNetActGELU(_SimplifiedLeNetAct):
    ACT_NAME = "gelu"


class LeNetActSoftLeakyReLU(_SimplifiedLeNetAct):
    ACT_NAME = "softlrelu"
