from torch.utils.data import Dataset
from torchvision import transforms
import torch
from torchvision.transforms.functional import to_pil_image

class MLPDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)
        elif not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)

        img = img.view(-1)  # Flatten for MLP input
        return img, torch.tensor(label, dtype=torch.long)


class CNNDataset(Dataset):
    def __init__(self, images, labels, common_transform=None, augment_transform=None):
        self.images = images
        self.labels = labels
        self.common_transform = common_transform
        self.augment_transform = augment_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]


        if isinstance(img, torch.Tensor):
            img = to_pil_image(img)
        elif not isinstance(img, Image.Image):
            img = Image.fromarray(img)


        if label == 1 and self.augment_transform:
            img = self.augment_transform(img)
        elif self.common_transform:
            img = self.common_transform(img)

        return img, torch.tensor(label, dtype=torch.long)
