import torch.nn as nn
import timm

class GazeModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=True
        )

        self.backbone.head = nn.Identity()

        # fine-tune last block only
        for name, p in self.backbone.named_parameters():
            if "blocks.11" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False

        self.regressor = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.regressor(feat)
