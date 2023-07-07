import torch
import os
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import albumentations as albu
import numpy as np
from torchvision import datasets

import cv2

def make_weights_for_balanced_classes(images, nclasses):
    n_images = len(images)
    count_per_class = [0] * nclasses
    for _, image_class in images:
        count_per_class[image_class] += 1
    weight_per_class = [0.] * nclasses
    for i in range(nclasses):
        weight_per_class[i] = float(n_images) / float(count_per_class[i])
    weights = [0] * n_images
    for idx, (image, image_class) in enumerate(images):
        weights[idx] = weight_per_class[image_class]
    return weights

def get_training_augmentation():

    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),

        albu.Blur(blur_limit=5, p=0.5),
        albu.RandomBrightnessContrast(brightness_limit=0.2,
                                      contrast_limit=0.2,
                                      p=0.5),
        albu.HueSaturationValue(hue_shift_limit=20,
                                sat_shift_limit=30,
                                p=0.5)
    ]
    composed_transform = albu.Compose(train_transform)

    return composed_transform

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')




def get_preprocessing():
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """

    _transform = [
        albu.Resize(512,512),
        albu.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
        albu.Lambda(image=to_tensor)
    ]
    return albu.Compose(_transform)


class ProstateDataset(torch.utils.data.Dataset):
    """Custom Dataset. Read images, apply augmentation and preprocessing transformations.
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)
    """
    def __init__(
            self,
            images_df,
            images_dir,
            augmentation=None,
            preprocessing=None
    ):
        self.ids = list(images_df.iloc[:,0].values)
        self.labels = np.argmax(images_df.iloc[:,1:5].values, axis=1)
        print(images_dir)
        print(self.ids)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.class_no = 4
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[i]


        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']

        return image, label, self.ids[i], self.ids[i]

    def __len__(self):
        return len(self.ids)


class CrowdProstateDataset(torch.utils.data.Dataset):
    """Custom Dataset. Read images, apply augmentation and preprocessing transformations.
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)
    """
    def __init__(
            self,
            images_df,
            images_dir,
            augmentation=None,
            preprocessing=None,
            experts=False
    ):
        self.ids = list(images_df["Patch filename"].values)
        self.labels = images_df.iloc[:,:7]
        self.MV_label = images_df["label"]
        self.EM_label = images_df["EM"]
        self.experts = experts
        if experts:
            self.GT_label = images_df["ground truth"]
        print(images_dir)
        print(self.ids)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.class_no = 4
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels.iloc[i,:].values
        MV_label = self.MV_label[i]
        EM_label = self.EM_label[i]
        if self.experts:
            GT_label = self.GT_label[i]


        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']
        
        if self.experts:
            return image, label, self.ids[i], MV_label, EM_label, GT_label
        else:
            return image, label, self.ids[i], MV_label, EM_label

    def __len__(self):
        return len(self.ids)

