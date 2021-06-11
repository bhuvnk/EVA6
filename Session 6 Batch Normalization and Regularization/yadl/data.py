from torchvision import datasets
import torch

from .transformations import get_train_test_transforms

def get_dataloaders(train_batch_size=None, val_batch_size=None, seed=42):
    train_transforms, test_transforms = get_train_test_transforms()

    train_ds = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)

    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    train_batch_size = train_batch_size or (128 if cuda else 64)
    val_batch_size = val_batch_size or (128 if cuda else 64)

    train_dataloader_args = dict(shuffle=True, batch_size=train_batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=train_batch_size)
    val_dataloader_args = dict(shuffle=True, batch_size=val_batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=val_batch_size)
    # dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

    train_loader = torch.utils.data.DataLoader(train_ds, **train_dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_ds, **val_dataloader_args)

    return train_loader, test_loader