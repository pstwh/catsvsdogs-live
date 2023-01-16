import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import mobilenet_v2


class CatsVsDogsMobileNet(nn.Module):
    def __init__(self):
        super(CatsVsDogsMobileNet, self).__init__()

        self.model = mobilenet_v2(weights="IMAGENET1K_V1")
        self.model.classifier[-1] = nn.Linear(
            in_features=1280, out_features=2, bias=True
        )

    def forward(self, x):
        return self.model(x)


class CatsVsDogsSimpleNet(nn.Module):
    def __init__(self):
        super(CatsVsDogsSimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=0
        )
        self.fc1 = nn.Linear(128 * 3 * 3, 2)

    # (n+2p-f)/s + 1
    def forward(self, x):
        x = self.conv1(x)  # (224 + 2 * 0 - 3) / 1 + 1 -> (1, 32, 222, 222)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # (1, 32, 222, 222) -> (1, 32, 111, 111)
        x = self.conv2(x)  # (111 + 2 * 0 - 3) / 1 + 1 -> (1, 64, 109, 109)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # (1, 64, 109, 109) -> (1, 64, 54, 54)
        x = self.conv3(x)  # (54 + 2 * 0 - 5) / 2 + 1
        x = F.relu(x)
        x = F.max_pool2d(x, 8)  # (1, 128, 12, 12)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return x

class CatsVsDogsAlexNet(nn.Module):
    def __init__(self, dropout: float = 0.5) :
        super().__init__()
        num_classes = 2
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    import torch

    model = CatsVsDogsSimpleNet()
    sample = torch.randn(1, 3, 224, 224, requires_grad=True)

    print(model(sample))
