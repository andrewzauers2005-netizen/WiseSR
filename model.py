import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        weights = models.VGG19_Weights.IMAGENET1K_V1

        vgg = models.vgg19(weights=weights).features[:35]
        vgg.eval()

        self.vgg = vgg
        self.vgg.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.vgg.eval()
        return self

    def forward(self, x):
        return self.vgg((x - self.mean) / self.std)


class FeatureEngine(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.vgg = VGGFeatureExtractor()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.main = nn.Sequential(
            nn.Conv2d(512, channels * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=channels * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Flatten(),

            nn.Linear(channels * 16, channels * 8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),

            nn.Linear(channels * 8, channels * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),

            nn.Linear(channels * 4, channels * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),

            nn.Linear(channels * 2, channels)
        )

        self.gamma = nn.Linear(channels, channels)
        self.beta = nn.Linear(channels, channels)

    def forward(self, x):
        x = self.pool(self.vgg(x))
        x = self.main(x)

        return self.gamma(x), self.beta(x)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=channels),
            nn.ReLU(),

            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=channels)
        )

    def forward(self, x):
        return x + self.block(x)


class WiseSR(nn.Module):
    def __init__(self, channels=64, n_blocks=16, n_tries=0):
        super().__init__()

        self.fe = FeatureEngine(channels)

        self.head = nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1)

        self.blocks = nn.ModuleList([
            ResBlock(channels) for _ in range(n_blocks)
        ])

        self.up = nn.Sequential(
            nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.n_tries = n_tries

    def forward(self, x):
        g, b = self.fe(x)

        g = g[:, :, None, None]
        b = b[:, :, None, None]

        for i in range(self.n_tries + 1):
            x = g * self.head(x) + b

            for block in self.blocks:
                x = F.relu(g * block(x) + b)

            x = self.up(F.relu(x))

            if i < self.n_tries:
                x = F.interpolate(x, scale_factor=0.5, mode='bilinear')

        return x
