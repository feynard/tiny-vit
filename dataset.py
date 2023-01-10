import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import RandomHorizontalFlip, RandomRotation

import pickle
from typing import Tuple


class CIFAR(Dataset):

    def __init__(self, path: str, flip_probability: float = 0.5, rotation_angle: int = 60):
        super().__init__()

        with open(path, 'rb') as f:
            data_dict = pickle.load(f)

        self.x = torch.tensor(data_dict['images'], dtype=torch.float32, requires_grad=False).permute(0, 3, 1, 2) / 255
        self.y = torch.tensor(data_dict['labels'], dtype=torch.long, requires_grad=False)

        self.size = len(self.x)

        self.augmentation = torch.nn.Sequential(
                RandomHorizontalFlip(flip_probability),
                RandomRotation(rotation_angle)
        )

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return self.augmentation(self.x[i]), self.y[i]


def create_loaders(
    file_path: str,
    batch_size: int,
    train_split: float = 0.8,
    flip_probability: float = 0.5,
    rotation_angle: int = 60) -> Tuple[DataLoader, DataLoader]:
    """
    file_path: path to pickled dictionary with keys
        'images': np.array of uint8 values of shape [N, H, W, 3]
        and 
        'labels': flattened np.array of uint8 values of length N
    
    train_split: ratio between 0 and 1, that is N * train_split ~ size of train dataset

    Returns train and validation dataloaders
    """

    dataset = CIFAR(file_path, flip_probability, rotation_angle)

    train_ds, validation_ds = torch.utils.data.random_split(
        dataset,
        (int(len(dataset) * train_split), len(dataset) - int(len(dataset) * train_split))
    )

    return \
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True), \
        DataLoader(validation_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
