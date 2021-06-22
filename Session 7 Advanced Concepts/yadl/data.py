import torchvision
import torch
import numpy as np
from .transformations import get_train_test_transforms

class Cifar10Dataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="./data", train=True, download=True, transform=None, viz=False):
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.viz = viz

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=np.array(image))
            image = transformed["image"]
            if self.viz:
              return image, label
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            image = torch.tensor(image, dtype=torch.float)
        return image, label

def get_dataloaders(train_batch_size=128, val_batch_size=128, seed=42):
    train_transforms, test_transforms = get_train_test_transforms()

    train_ds = Cifar10Dataset('./data', train=True, download=True, transform=train_transforms)
    test_ds = Cifar10Dataset('./data', train=False, download=True, transform=test_transforms)

    cuda = torch.cuda.is_available()
#     print("CUDA Available?", cuda)

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    # train_batch_size = train_batch_size or (128 if cuda else 64)
    # val_batch_size = val_batch_size or (128 if cuda else 64)

    train_dataloader_args = dict(shuffle=True, batch_size=train_batch_size, num_workers=4, pin_memory=True)
    val_dataloader_args = dict(shuffle=True, batch_size=val_batch_size, num_workers=4, pin_memory=True)
    # dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

    train_loader = torch.utils.data.DataLoader(train_ds, **train_dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_ds, **val_dataloader_args)

    return train_loader, test_loader
