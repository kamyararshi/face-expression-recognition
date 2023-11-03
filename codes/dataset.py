## dataset classes for FER and CK+

from __future__ import print_function
from PIL import Image
import numpy as np
from pathlib import Path
import os
import glob
import ast


import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from sklearn.model_selection import train_test_split



EMOTIONS_CKPlus = ['neutral', 'anger', 'contempt',
                   'disgust', 'fear', 'happy',
                   'sadness', 'surprise']
EMOTIONS_Emotic = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection',
                   'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear',
                   'Happiness', 'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']


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
            image: Tensor of shape (C, H, W) and type torch.float32 (already normalized)
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



class Emotic_CSVDataset(Dataset):
    ''' Custom Emotic dataset class. Use csv files and generated data at runtime. '''
    def __init__(self, data_df, cat2ind, context_norm, body_norm, transform=True, data_src = './dataset/Emotic/'):
        super(Emotic_CSVDataset,self).__init__()
        self.data_df = data_df
        self.data_src = data_src 
        self.transform = transform 
        self.cat2ind = cat2ind
        self.context_norm = transforms.Normalize(mean=context_norm[0], std=context_norm[1])  # Normalizing the context image with context mean and context std
        self.body_norm = transforms.Normalize(mean=body_norm[0], std=body_norm[1])           # Normalizing the body image with body mean and body std

    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, index):
        row = self.data_df.loc[index]
        image_context = Image.open(os.path.join(self.data_src, "emotic", row['Folder'], row['Filename']))
        bbox = ast.literal_eval(row['BBox'])
        image_body = image_context.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        
        cat_labels = ast.literal_eval(row['Categorical_Labels'])
        cont_labels = ast.literal_eval(row['Continuous_Labels'])
        one_hot_cat_labels = self.cat_to_one_hot(cat_labels)

        # Transformations and Augmentaions
        totensor = transforms.Compose(
            [
                transforms.ToTensor(),
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
        
        
        image_context = image_context.resize((224, 224))
        image_body = image_body.resize((128, 128))
        image_context_tensor = totensor(image_context).to(torch.float32)
        image_body_tensor = totensor(image_body).to(torch.float32)
        
        # Check if the image has 3 channels
        image_context_tensor, image_body_tensor = self._ensure_3_channels(image_context_tensor, image_body_tensor)

        assert image_body_tensor.shape[0] == image_context_tensor.shape[0] == 3, print("#####error:\n", image_body_tensor.shape, image_context_tensor.shape, row['Folder'], row['Filename'],"#####")

        if self.transform:
            # image context
            image_context = self.context_norm(image_context_tensor)
            image_context = applier(image_context)
            # image body
            image_body = self.body_norm(image_body_tensor)
            image_body = applier(image_body)
            
            
        else:
            
            image_context = self.context_norm(image_context)
            image_body = self.body_norm(image_body_tensor)
            

        return image_context, image_body, torch.tensor(one_hot_cat_labels, dtype=torch.float32), torch.tensor(cont_labels[0], dtype=torch.long)
    
    def cat_to_one_hot(self, cat):
        one_hot_cat = np.zeros(26)
        for em in cat:
            one_hot_cat[self.cat2ind[em]] = 1
        return one_hot_cat

    def getitem_2plot(self, index):

        row = self.data_df.loc[index]
        image_context = Image.open(os.path.join(self.data_src, "emotic", row['Folder'], row['Filename']))
        bbox = ast.literal_eval(row['BBox'])
        image_body = image_context.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

        return image_context, image_body
    
    def _ensure_3_channels(self, image_context, image_body):
        """ensure 3 channels for an image if it has other than 3 number of channels
        """
        if image_context.size(0) == 4 or image_body.size(0) == 4:
            image_context = image_context[:3, ...]
            image_body = image_body[:3, ...]
        if image_context.size(0) == 1:
            image_context = image_context.repeat(3, 1, 1)
        if image_body.size(0) == 1:
            image_body = image_body.repeat(3,1,1)
        
        
        return image_context, image_body