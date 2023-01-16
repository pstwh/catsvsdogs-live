from torchvision.models import mobilenet_v2
import torch.nn as nn


class CatsVsDogsMobileNet(nn.Module):
    def __init__(self):
        super(CatsVsDogsMobileNet, self).__init__()

        self.model = mobilenet_v2(weights="IMAGENET1K_V1")
        self.model.classifier[-1] = nn.Linear(
            in_features=1280, out_features=2, bias=True
        )

    def forward(self, x):
        return self.model(x)
