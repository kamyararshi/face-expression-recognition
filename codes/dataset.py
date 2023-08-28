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
from sklearn.model_selection import train_test_split



EMOTIONS_CKPlus = ['neutral', 'anger', 'contempt',
                   'disgust', 'fear', 'happy',
                   'sadness', 'surprise']

class CKPlusDataset(Dataset):

    _ext_imgs = ".png"
    _ext_labels = ".txt"
    _images_folder = "cohn-kanade-images"
    _labels_folder = "Emotion"

    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
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

        """
        label_files = glob.glob(os.path.join(self.emotion_dir, '*', '*', '*'+self._ext_labels))
        image_files = [labels_file.replace(self._labels_folder, self._images_folder).replace('_emotion'+self._ext_labels, self._ext_imgs) for labels_file in label_files]
        return image_files, label_files

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image_path, emotion_path = self.image_path[idx], self.emotion_path[idx]
        
        image = Image.open(image_path).convert('RGB')
        
        with open(emotion_path, 'r') as f:
            emotion_label = int(f.readline().strip().split('.')[0])
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(emotion_label, dtype=torch.float32)

    






# From https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch/blob/master/CK.py
class CK(Dataset):
    """`CK+ Dataset.

    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``

        there are 135,177,75,207,84,249,54 images in data
        we choose 123,159,66,186,75,225,48 images for training
        we choose 12,8,9,21,9,24,6 images for testing
        the split are in order according to the fold number
    """

    _ext_imgs = ".png"
    _ext_labels = ".txt"

    def __init__(self, path, split='Training', fold = 1, transform=None):
        self.transform = transform
        self.split = split  # training set or test set
        self.fold = fold # the k-fold cross validation
        self.data = h5py.File('./data/CK_data.h5', 'r', driver='core')

        number = len(self.data['data_label']) #981
        sum_number = [0,135,312,387,594,678,927,981] # the sum of class number
        test_number = [12,18,9,21,9,24,6] # the number of each class

        test_index = []
        train_index = []

        for j in xrange(len(test_number)):
            for k in xrange(test_number[j]):
                if self.fold != 10: #the last fold start from the last element
                    test_index.append(sum_number[j]+(self.fold-1)*test_number[j]+k)
                else:
                    test_index.append(sum_number[j+1]-1-k)

        for i in xrange(number):
            if i not in test_index:
                train_index.append(i)

        print(len(train_index),len(test_index))

        # now load the picked numpy arrays
        if self.split == 'Training':
            self.train_data = []
            self.train_labels = []
            for ind in xrange(len(train_index)):
                self.train_data.append(self.data['data_pixel'][train_index[ind]])
                self.train_labels.append(self.data['data_label'][train_index[ind]])

        elif self.split == 'Testing':
            self.test_data = []
            self.test_labels = []
            for ind in xrange(len(test_index)):
                self.test_data.append(self.data['data_pixel'][test_index[ind]])
                self.test_labels.append(self.data['data_label'][test_index[ind]])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == 'Testing':
            img, target = self.test_data[index], self.test_labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        elif self.split == 'Testing':
            return len(self.test_data)
