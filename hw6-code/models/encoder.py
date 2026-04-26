import torch
import torchvision.models as models
from torch import Tensor, nn


class Encoder(nn.Module):
    """Encoder model.

    Supported backbones:
        - "resnet50": torchvision ResNet-50, output (B, 49, 2048).
        - "efficientnet_v2_s": torchvision EfficientNet-V2-S, output (B, 49, 1280).
    """

    def __init__(
        self,
        encoded_size: tuple[int, int] = (7, 7),
        finetune: bool = False,
        backbone: str = "resnet50",
    ):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            # 去掉最后两层 (avgpool, fc)，保留卷积特征
            self.backbone = nn.Sequential(*list(net.children())[:-2])
            self._finetune_from = 5
        elif backbone == "efficientnet_v2_s":
            net = models.efficientnet_v2_s(
                weights=models.EfficientNet_V2_S_Weights.DEFAULT
            )
            self.backbone = net.features
            self._finetune_from = 5
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.pool = nn.AdaptiveAvgPool2d(encoded_size)
        self.finetune(finetune)

    def forward(self, images: Tensor):
        """Extract image features.

        Args:
            images (Tensor): Input images with shape (batch, channels, height, width).

        Returns:
            Tensor: Extracted features with shape (batch, num_pixels, feature_dim).
        """
        x: Tensor = self.backbone(images)
        x = self.pool(x)
        x = x.permute(0, 2, 3, 1).flatten(1, 2)
        return x

    def finetune(self, finetune: bool = True):
        self.backbone.requires_grad_(False)
        for child in list(self.backbone.children())[self._finetune_from :]:
            child.requires_grad_(finetune)
