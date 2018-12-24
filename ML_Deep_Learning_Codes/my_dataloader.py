#Reference Link : This Blog : https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
import numpy as np
import cv2
import keras

training_data_folder = '/scratch/bam_subset_2_0/'

class DataGenerator(keras.utils.Sequence):
    #'Generates data for Keras'

    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1, n_classes=10, shuffle=True):

        #'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        #'Denotes the number of batches per epoch'
        return int((np.floor(len(self.list_IDs)) / self.batch_size))

    def __getitem__(self, index):
        
        #'Generate one batch of data'
        #Generates indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        #find list of ids
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate Data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):

        #Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):

        #Generates data containing batch_size samples : X : (n_samples, *dim, n_channels)
        #Initialization

        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype = int)

        #Generate Data

        for i, ID in enumerate(list_IDs_temp):
            #Store sample
            #X[i,] = np.load('data/'+ID+'.npy')
            image_size = self.dim[0]
            image = cv2.imread(training_data_folder+ID+'.jpg')
            image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0/255.0) 
            X[i,] = image

            #Store Class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes = self.n_classes)

