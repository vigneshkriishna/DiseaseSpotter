import torch.nn as nn
import torch
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return x * self.sigmoid(attention)

class EfficientNetWithAttention(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetWithAttention, self).__init__()
        weights = EfficientNet_B3_Weights.DEFAULT
        self.base_model = efficientnet_b3(weights=weights)
        self.spatial_attention = SpatialAttention()
        self.base_model.features[-1].add_module("spatial_attention", self.spatial_attention)
        self.num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        if not self.training:
            self.base_model.classifier[1].track_running_stats = False
        return self.base_model(x)
