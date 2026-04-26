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
        # Wh/Wf map hidden state and image features into the same space.
        self.Wf = nn.Linear(feature_dim, hidden_dim)
        self.Wh = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, features: Tensor, hidden: Tensor):
        """Forward pass of AdditiveAttention.

        Args:
            features (Tensor): Input features with shape (batch, num_pixels, feature_dim).
            hidden (Tensor): Hidden state with shape (batch, hidden_dim).

        Returns:
            Tensor: Attention output with shape (batch, feature_dim).
        """
        # e = v * tanh(Wf * features + Wh * hidden)
        # alpha = softmax(e)
        # context = sum(alpha * features)
        feature_attn = self.Wf(features)
        hidden_attn = self.Wh(hidden).unsqueeze(1)
        e = self.v(torch.tanh(feature_attn + hidden_attn)).squeeze(-1)
        alpha = torch.softmax(e, dim=1).unsqueeze(-1)
        context = torch.sum(alpha * features, dim=1)
        return context
