import torch
from torch import Tensor, nn


class AdditiveAttention(nn.Module):
    """Additive attention in Show, Attend, and Tell."""

    def __init__(self, feature_dim: int, hidden_dim: int):
        """Initialize AdditiveAttention.

        Args:
            feature_dim (int): Feature dimension.
            hidden_dim (int): Hidden dimension.
        """
        super().__init__()
        # ======== TODO: Init linear layers for attention ========
        # 1. 实现 Wh, Wf, 用于映射隐藏状态和特征
        # 2. 实现 v, 用于进行输出的线性映射
        # ========================================================

    def forward(self, features: Tensor, hidden: Tensor):
        """Forward pass of AdditiveAttention.

        Args:
            features (Tensor): Input features with shape (batch, num_pixels, feature_dim).
            hidden (Tensor): Hidden state with shape (batch, hidden_dim).

        Returns:
            Tensor: Attention output with shape (batch, num_pixels, hidden_dim).
        """
        # ============ TODO: forward pass ============
        # e = v * tanh(Wf * features + Wh * hidden)
        # alpha = softmax(e)
        # context = sum(alpha * features)
        raise NotImplementedError
        # ============================================
        return context
