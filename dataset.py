import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
import os


class SRDataset(Dataset):
    def __init__(self, path, img_size=64):
        self.lr_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])

        self.hr_transform = transforms.Compose([
            transforms.Resize(img_size * 2),
            transforms.CenterCrop(img_size * 2),
            transforms.ToTensor()
        ])

        self.data = []

        for subset in os.listdir(path):
            subset_path = os.path.join(path, subset)

            if not os.path.isdir(subset_path):
                continue

            imgs = [
                os.path.join(subset_path, img)
                for img in os.listdir(subset_path)
                if img.lower().endswith(('.jpg', '.png', '.jpeg'))
            ]

            self.data.extend(imgs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        while True:
            try:
                image = Image.open(self.data[item]).convert('RGB')
                return self.lr_transform(image), self.hr_transform(image)

            except (PIL.UnidentifiedImageError, Image.DecompressionBombError, OSError):
                item = random.randint(0, self.__len__())
