import cv2
import torchvision
import torch
import torchvision.transforms as transforms

# Albumentations for augmentations

import albumentations as A
from albumentations.pytorch import ToTensorV2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class Data_Set(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label

train_transforms = A.Compose(

    [
        A.PadIfNeeded(min_height = 36,min_width = 36,p = 1),
        A.RandomCrop(height=32,width = 32),
        A.HorizontalFlip(p = 0.5),
        A.Cutout(num_holes = 1,max_h_size = 8 , max_w_size = 8 , fill_value = 0 , p = 0.5),
        #A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.3),
        #A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16,fill_value=0.4734,mask_fill_value = None,p=0.3),
        A.Normalize(mean = (0.4914, 0.4822, 0.4465),std = (0.2470, 0.2435, 0.2616),p =1.0),
        ToTensorV2()
    ], p = 1.0

)

test_transforms = A.Compose(

    [
        #A.HorizontalFlip(p=0.5),
        #A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        #A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16,fill_value=0.4734),
        A.Normalize(mean = (0.4914, 0.4822, 0.4465),std = (0.2470, 0.2435, 0.2616),p=1.0),
        ToTensorV2()
    ], p = 1.0

)