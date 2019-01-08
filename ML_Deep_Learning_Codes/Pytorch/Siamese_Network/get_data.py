from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import csv
import cv2
import random

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

def parse_byte(b):
    if isinstance(b, str):
        return ord(b)
    return b

def read_labels_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        labels = [parse_byte(b) for b in data[8:]]
        assert len(labels) == length
        return labels
        return torch.LongTensor(labels)


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])

        print(length, num_rows, num_cols)

        images = []
        idx = 16
        for l in range(length):
            img = []
            images.append(img)
            for r in range(num_rows):
                row = []
                img.append(row)
                for c in range(num_cols):
                    row.append(parse_byte(data[idx]))
                    idx += 1
        return images
        
        assert len(images) == length
        return torch.ByteTensor(images).view(-1, 28, 28)

if __name__ == "__main__":
    
    out = read_image_file('data/mnist/t10k-images.idx3-ubyte')
    print("Image file reading complete")
    ''' 
    for i, img in enumerate(out):
        print ("Writing image", i)
        img = np.asarray(img, dtype=np.uint8)
        cv2.imwrite('data/mnist/testing/'+str(i)+'.jpg', img)
    '''
    print("Labels file reading complete")
    labels_train = read_labels_file('data/mnist/train-labels.idx1-ubyte')
    print(labels_train[:10])

    classes = [0,1,2,3,4,5,6,7,8,9]
    classes_dict = {}
    for idx, label in enumerate(labels_train):
        if label not in classes_dict:
            classes_dict[label] = []
        classes_dict[label].append(idx)

    #writing training_triplet_filename
    
    f = open('training_triplet_filename.txt', 'w')
    for i in range(len(labels_train)):
        print("writing Training Triplet Filename", i)
        a = i
        b = -1
        c = -1
        #Similar candidate
        while True:
            c = random.randint(0, len(classes_dict[labels_train[i]])-1)
            if c != a:
                c = classes_dict[labels_train[i]][c]
                break
        #dissimilar candidate
        while True:
            x = random.randint(0, 9)
            if x != labels_train[i]:
                b = random.randint(0, len(classes_dict[x])-1)
                b = classes_dict[x][b]
                break
        f.write(str(a)+' '+str(b)+' '+str(c)+'\n')

    f.close()
    print("writing Training Triplet Filename Complete")

    exit()
