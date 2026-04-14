import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, use_pretrained=True):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        weights = ResNet50_Weights.IMAGENET1K_V2 if use_pretrained else None
        model = resnet50(weights=weights).eval().to(self.device)

        # self.preprocess = weights.transforms() if weights is not None else None
        self.feature_extractor = nn.Sequential(*list(model.children())[:-2])

    def forward(self, x):
        """
        x: tensor of shape [B, 3, H, W]
        returns: feature maps of shape [B, 2048, H/32, W/32]
        """
        x = x.to(self.device)
        with torch.no_grad():
            return self.feature_extractor(x)






