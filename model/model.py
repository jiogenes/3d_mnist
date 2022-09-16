import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.feature_extract = nn.Sequential(
            nn.Conv3d(1, 8, 3), # (B, 8, 14, 14, 14)
            nn.ReLU(inplace=True),

            nn.Conv3d(8, 16, 3), # (B, 16, 12, 12, 12)
            nn.ReLU(inplace=True),
            
            nn.MaxPool3d(2), # (B, 16, 6, 6, 6)
            nn.Dropout3d(0.2),

            nn.Conv3d(16, 32, 3), # (B, 32, 4, 4, 4)
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 64, 3), # (B, 64, 2, 2, 2)
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(2), # (B, 64, 1, 1, 1)
            nn.Dropout(0.2),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(512, 10),
        )
        
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1), # (16, 8, 16, 16, 16)
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2), # (16, 8, 8, 8, 8)

            nn.Conv3d(8, 32, kernel_size=3), # (16, 32, 6, 6, 6)
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2), # (16, 32, 3, 3, 3)
            nn.Flatten()
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 3 * 3 * 3, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 4 * 3)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.eye(3, 4).view(1, -1).squeeze())

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 3, 4)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

    def forward(self,x):
        # x = self.stn(x)
        
        x = self.feature_extract(x)
        x = self.classifier(x)
        return x