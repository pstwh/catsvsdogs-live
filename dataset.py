import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class CatVsDogsDataset(Dataset):
    def __init__(self, images, classes=[], transform=None):
        self.images = images
        self.classes = classes
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.images[index]
        image = np.array(Image.open(image_path).convert("RGB"))

        if self.transform:
            X = self.transform(image=image)["image"]

        y = self.classes.index(image_path.split("/")[-2])

        return X, y, image_path

    def __len__(self):
        return len(self.images)
