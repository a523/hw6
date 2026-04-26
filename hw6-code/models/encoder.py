import torch
import torchvision.models as models
from torch import Tensor, nn
from torchvision.models import ResNet50_Weights


class Encoder(nn.Module):
    """Encoder model."""

    def __init__(self, encoded_size=(7, 7), finetune=False):
        super().__init__()
        # 加载预训练 resnet
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # 移除最后两层（平均池化和全连接层）
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        # 自适应池化到固定尺寸
        # self.pool = nn.AdaptiveAvgPool2d(encoded_size)
        # 默认不微调
        self.finetune(finetune)

    def forward(self, images: Tensor):
        """Extract image features.

        Args:
            images (Tensor): Input images with shape (batch, channels, height, width).

        Returns:
            Tensor: Extracted features with shape (batch, num_pixels, feature_dim).
        """
        # 提取特征
        x: Tensor = self.resnet(images)
        # 池化到固定尺寸
        # x: Tensor = self.pool(x)
        # B C H W -> B (H W) C
        x = x.permute(0, 2, 3, 1).flatten(1, 2)
        return x

    def finetune(self, finetune=True):
        # 冻结所有参数
        self.resnet.requires_grad_(False)
        # 仅微调后几层
        for child in list(self.resnet.children())[5:]:
            child.requires_grad_(finetune)
