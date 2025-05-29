import torch.nn as nn
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


# Custom Retina cls head with dropout
class RetinaClassificationHeadDropout(RetinaNetClassificationHead):
    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        num_classes: int,
        prior_probability: float = 0.01,
        norm_layer=None,
        dropout: float = 0.1,
    ):
        super().__init__(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
            prior_probability=prior_probability,
            norm_layer=norm_layer,
        )
        # Insert Dropout after each Conv2dNormActivation module in self.conv
        new_conv_layers = []
        for layer in self.conv:
            new_conv_layers.append(layer)
            new_conv_layers.append(nn.Dropout(p=dropout))
        self.conv = nn.Sequential(*new_conv_layers)
