# Partly written by author: Siddhartha Gairola
# Substantially adaptee from References 1, 2 in Readme.txt file.

from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

#Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion() #interative
print("Import Successful")

def default_image_loader(path):
    return Image.open(path).convert('RGB')

class TripletImageLoader(Dataset):

    def __init__(self, base_path, filenames_filename, triplets_filename, transform=None, loader = default_image_loader):

        # filenames_filename => A text file with each line containing a path to an image, e.g., images/class1/sample.jpg
        # triplets_filename => A text file with each line containing 3 integers, where integer i refers to the i-th image
        # in filenames_filename. For a line with integers "a b c", a triplet is defined such that image a is more similar
        # to image c than it is to image b.

        self.base_path = base_path
        self.filenamelist = []

        for line in open(filenames_filename):
            self.filenamelist.append(line.rstrip('\n'))

        triplets = []

        for line in open(triplets_filename):
            triplets.append((line.split()[0], line.split()[1], line.split()[2])) # anchor, far, close

        self.triplets = triplets
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path1, path2, path3 = self.triplets[index]
        img1 = self.loader(os.path.join(self.base_path,self.filenamelist[int(path1)]))
        img2 = self.loader(os.path.join(self.base_path,self.filenamelist[int(path2)])) #far => negative
        img3 = self.loader(os.path.join(self.base_path,self.filenamelist[int(path3)])) #close => positive

        if self.tranform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3

    def __len__(self):
        return len(self.triplets)

