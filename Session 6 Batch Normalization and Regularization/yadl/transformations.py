from torchvision import transforms


def get_train_test_transforms():
    train_transforms = transforms.Compose([
        transforms.RandomAffine((-6.9, 6.9), translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return train_transforms, test_transforms
