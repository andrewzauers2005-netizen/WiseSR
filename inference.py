from dataset import SRDataset
from model import WiseSR

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

dataset = SRDataset(path='test', img_size=64)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

model = WiseSR()
checkpoint = torch.load('trained_models/wise_sr_epoch_1.pth', map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()

criterion = nn.L1Loss()

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'{params:,} trainable parameters')

for pair in dataloader:
    lr_img = pair[0].squeeze(0).permute(1, 2, 0)
    hr_img = pair[1].squeeze(0).permute(1, 2, 0)

    up_img = F.interpolate(pair[0], scale_factor=2, mode='bilinear').squeeze(0).permute(1, 2, 0)

    with torch.no_grad():
        nn_img = model(pair[0]).detach().squeeze(0).permute(1, 2, 0)

    # print(f'Loss_up: {criterion(up_img, hr_img):.4f}')
    # print(f'Loss_nn: {criterion(nn_img, hr_img):.4f}')
    # print()

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(lr_img)
    axes[0, 0].set_title('low_res')

    axes[0, 1].imshow(hr_img)
    axes[0, 1].set_title('high_res (target)')

    axes[1, 0].imshow(up_img)
    axes[1, 0].set_title('upscaled')

    axes[1, 1].imshow(nn_img)
    axes[1, 1].set_title('output')

    plt.show()
