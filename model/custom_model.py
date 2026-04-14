import torch
import torch.nn as nn


class ObjectPointNetRegressor(nn.Module):
    """
    One sample = one object point cloud.
    Input:
        points: (B, N, C) where C=3 for XYZ or C=6 for XYZRGB
    Output:
        pred_bbox: (B, 8, 3)  # predict 8 box corners directly
    """

    def __init__(self, input_channels: int = 3, dropout: float = 0.3):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(256, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(128, 24),   # 8 corners * 3 coords
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        points: (B, N, C)
        returns: (B, 8, 3)
        """
        if points.ndim != 3:
            raise ValueError(f"Expected points with shape (B, N, C), got {points.shape}")

        x = points.transpose(1, 2)      # (B, C, N)
        x = self.feature_extractor(x)   # (B, 512, N)
        x = torch.max(x, dim=2)[0]      # (B, 512)
        x = self.regressor(x)           # (B, 24)
        x = x.view(-1, 8, 3)            # (B, 8, 3)
        return x



