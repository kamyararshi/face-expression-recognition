## dataset classes for FER and CK+

from __future__ import print_function
from PIL import Image
import numpy as np
from pathlib import Path
import os
import glob


import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from sklearn.model_selection import train_test_split



EMOTIONS_CKPlus = ['neutral', 'anger', 'contempt',
                   'disgust', 'fear', 'happy',
                   'sadness', 'surprise']

class CKPlusDataset(Dataset):

    _ext_imgs = ".png"
    _ext_labels = ".txt"
    _images_folder = "cohn-kanade-images"
    _labels_folder = "Emotion"

    def __init__(self, root_dir, transform=True, train=True, device='cpu'):
        """`CK+ Dataset.

        Args:
            train (bool, optional): If True, creates dataset from training set, otherwise
                creates from test set.
            transform (bool, optional): Whether to do augmentation or not on the dataset
            device (string): cuda or cpu
        """
        self.root_dir = root_dir
        self.transform = transform
        self.device = device
        self.image_dir = os.path.join(self.root_dir, self._images_folder)
        self.emotion_dir = os.path.join(self.root_dir, self._labels_folder)
        self.all_samples = self._load_samples()
        imgs_train, imgs_test, labels_train, labels_test = train_test_split(
            self.all_samples[0], self.all_samples[1], test_size=.1, random_state=2023
            )

        if train:
            self.image_path, self.emotion_path = imgs_train, labels_train
        else:
            self.image_path, self.emotion_path = imgs_test, labels_test

    def _load_samples(self):
        """
        Returns images and labels path
            image_files: path of image files 
            label: path of corressponding emotion labels

        """
        label_files = glob.glob(os.path.join(self.emotion_dir, '*', '*', '*'+self._ext_labels))
        image_files = [labels_file.replace(self._labels_folder, self._images_folder).replace('_emotion'+self._ext_labels, self._ext_imgs) for labels_file in label_files]
        return image_files, label_files

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        """
        Returns images its labels w.r.t given index 
            image: Tensor of shape (C, H, W) and type torch.float32
            label: labels scalar of type torch.long

        """
        image_path, emotion_path = self.image_path[idx], self.emotion_path[idx]
        
        image = Image.open(image_path).convert('RGB')
        
        with open(emotion_path, 'r') as f:
            emotion_label = int(f.readline().strip().split('.')[0])
            emotion_label = torch.tensor(emotion_label, dtype=torch.long)
        
        # Transformations and Augmentaions
        totensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize([490, 640]),
            ]
        )
        applier = transforms.RandomApply(
                    transforms=[
                        transforms.RandomAffine(degrees=30, translate=(.05, .1), scale=(1.1, 1.3)),
                        transforms.ColorJitter(brightness=.5, hue=.3, contrast=.3, saturation=.2),
                        transforms.GaussianBlur(kernel_size=(5,7), sigma=(.1,5)),
                        transforms.RandomHorizontalFlip(p=.6),
                    ],
                    p=0.5
                )
        if self.transform:
            image = totensor(image).to(torch.float32)
            image = applier(image)
            
        else:
            image = totensor(image).to(torch.float32)
        
        return image.to(device=self.device), emotion_label.to(device=self.device)

