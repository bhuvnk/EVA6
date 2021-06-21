from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_test_transforms():
    train_transforms = A.Compose(
        [
            A.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
            A.HorizontalFlip(),
            A.ShiftScaleRotate(),
            A.CoarseDropout(1, 16, 16, 1, 16, 16,fill_value=0.473363),
            A.ToGray()
#             ToTensorV2(),
        ])

    val_transforms = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#             ToTensorV2(),
        ])

    return train_transforms, val_transforms
