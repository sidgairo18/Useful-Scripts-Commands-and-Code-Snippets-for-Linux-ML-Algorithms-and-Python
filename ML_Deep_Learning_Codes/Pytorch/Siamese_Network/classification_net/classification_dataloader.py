# Partly written by author: Siddhartha Gairola
# Substantially adaptee from References 1, 2 in Readme.txt file.

from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from PIL import Image

#Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion() #interative
print("Import Successful TripletImageLoader")

def default_image_loader(path):
    return (Image.open(path).convert('RGB')).resize((224, 224), Image.ANTIALIAS)
    #return Image.open(path)

class ClassificationImageLoader(Dataset):

    def __init__(self, base_path, filenames_filename, labels_filename, transform=None, loader = default_image_loader):

        # filenames_filename => A text file with each line containing a path to an image, e.g., images/class1/sample.jpg
        # labels_filename => A text file with each line containing 1 integer, label index of the image.

        self.base_path = base_path
        self.filenamelist = []

        for line in open(filenames_filename):
            self.filenamelist.append(line.rstrip('\n'))

        labels = []

        for line in open(labels_filename):
            labels.append(int(line.strip())) # label

        self.labels = labels
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        #print((os.path.join(self.base_path,self.filenamelist[int(path1)])))
        img1 = self.loader(os.path.join(self.base_path,self.filenamelist[index]))

        if self.transform:
            img1 = self.transform(img1)

        return img1, self.labels[index]

    def __len__(self):
        return len(self.labels)

