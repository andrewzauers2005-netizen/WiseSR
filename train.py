from dataset import SRDataset
from model import WiseSR

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings('ignore')
os.makedirs('trained_models', exist_ok=True)

dataset = SRDataset(path='../Images-1/images')
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = WiseSR(n_tries=0)

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    progress = tqdm(dataloader, desc=f'Epoch {epoch}')
    losses = []

    for i, (lr, hr) in enumerate(progress):
        optimizer.zero_grad()

        pred = model(lr)
        loss = criterion(pred, hr)

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        losses.append(loss.item())

        progress.set_postfix({
            'loss': f'{losses[-1]:.4f}',
            'avg_loss': f'{sum(losses) / len(losses):.4f}'
        })

        if i % 10 == 0:
            torch.save(
                model.state_dict(),
                f'trained_models/wise_sr_epoch_{epoch}.pth'
            )
