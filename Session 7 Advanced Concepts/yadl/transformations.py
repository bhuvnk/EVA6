from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_test_transforms():
    train_transforms = A.Compose(
      [
        A.HorizontalFlip(),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.CoarseDropout(max_holes = 1,
                       max_height= 16,
                       max_width= 16,
                       min_holes = 1,
                       min_height= 16,
                       min_width= 16),
          
        A.ToGray(0.1),
          
        A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
        ToTensorV2(),
      ])

    val_transforms = A.Compose(
        [
          A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
          ToTensorV2(),
        ])

    return train_transforms, val_transforms
