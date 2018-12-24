#A shallow CNN Siamese Network with Triplet Loss.

#!/usr/bin/python                                                       
# Author: Siddhartha Gairola (siddhartha dot gairola at iiit dot ac dot in)
from __future__ import division                                         
import tensorflow as tf                                                 
tf.set_random_seed(1)                                                   
                                                                        
from keras.models import Model, load_model                              
from keras.layers import BatchNormalization, Activation, Dense, Dropout, Flatten, Input, Lambda
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D            
from keras.regularizers import l2                                       
from keras.layers.merge import dot, multiply                            
from keras.callbacks import TensorBoard, ModelCheckpoint                
from keras import optimizers                                            
import os                                                               
import pickle                                                           
import pdb                                                              
import json                                                             
import pickle as pkl                                                    
import numpy as np                                                      
from tqdm import tqdm                                                   
import random                                                           
#from my_dataloader_for_triplet import DataGenerator                    
from my_dataloader_for_triplet import DataGenerator                                 
                                                                        
#Alpha : The Triplet Loss Parameter                                     
                                                                        
def triplet_loss(x, ALPHA=0.2):                                         
                                                                        
    anchor, positive, negative = x                                      
                                                                        
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
                                                                        
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), ALPHA)         
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)               
                                                                        
    return loss                                                         
                                                                        
class cnNet():                                                        
                                                                        
    def __init__(self, input_shape_x, input_shape_y, input_shape_z, n_classes, reg_lambda):
                                                                        
        self.input_shape_x = input_shape_x                              
        self.input_shape_y = input_shape_y                              
        self.input_shape_z = input_shape_z                              
        self.n_classes = n_classes                                      
        self.reg_lambda = reg_lambda                                    
                                                                        
                                                                        
    def create_model(self):                                             
                                                                        
        anchor_example = Input(shape=(self.input_shape_x, self.input_shape_y, self.input_shape_z), name='input_1')
        positive_example = Input(shape=(self.input_shape_x, self.input_shape_y, self.input_shape_z), name='input_2')
        negative_example = Input(shape=(self.input_shape_x, self.input_shape_y, self.input_shape_z), name='input_3')

        input_image = Input(shape=(self.input_shape_x, self.input_shape_y, self.input_shape_z))
                                                                        
        conv1 = Conv2D(32, kernel_size=(5, 5), strides=4, padding='same', kernel_regularizer=l2(self.reg_lambda), name='conv1')(input_image)
        norm1 = BatchNormalization(name='norm1')(conv1)                 
        relu1 = Activation('relu', name = 'relu1')(norm1)               
        pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(relu1)     
                                                                        
        conv2 = Conv2D(32, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(self.reg_lambda), name='conv2')(pool1)
        norm2 = BatchNormalization(name='norm2')(conv2)                 
        relu2 = Activation('relu', name = 'relu2')(norm2)               
        pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(relu2)     
                                                                        
        padding3 = ZeroPadding2D((1, 1))(pool2)                         
        conv3 = Conv2D(64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(self.reg_lambda), name='conv3')(padding3)
        norm3 = BatchNormalization(name='norm3')(conv3)                 
        relu3 = Activation('relu', name='relu3')(norm3)                 
        pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(relu3)     
        
        flatten4 = Flatten()(pool3)                                     
        fc4 = Dense(128, kernel_regularizer=l2(self.reg_lambda), name='fc4')(flatten4)
        norm4 = BatchNormalization(name='norm4')(fc4)                   
        relu4 = Activation('relu', name='relu4')(norm4)                 
        drop4 = Dropout(0.5)(relu4)                                     
        
                                                                        
        ##############Adding the Bottleneck layer Here#######################################################
        bottleneck_layer = Dense(64, kernel_regularizer=l2(self.reg_lambda), name='bottleneck_layer')(drop4)
        bottleneck_norm = BatchNormalization(name='bottleneck_norm')(bottleneck_layer)
                                                                        
        fin = Dense(self.n_classes)(bottleneck_norm)                    
        fin_norm = BatchNormalization(name='fin_norm')(fin)             
        fin_softmax = Activation('softmax')(fin_norm)                   
        ######################################################################################################
        
        ###########Triplet Model Which learns the embedding layer relu6####################
        self.triplet_model = Model(input_image, bottleneck_norm)                  
        positive_embedding = self.triplet_model(positive_example)       
        negative_embedding = self.triplet_model(negative_example)       
        anchor_embedding = self.triplet_model(anchor_example)           
        ###########Triplet Model Which learns the embedding layer relu6####################
        
        #The Triplet Model which optimizes over the triplet loss.       
        loss = Lambda(triplet_loss, output_shape=(1,))([anchor_embedding, positive_embedding, negative_embedding])
        self.triplet_model_worker = Model(inputs=[anchor_example, positive_example, negative_example], outputs = loss)
                                                                        
        adam_opt = optimizers.Adam(lr=0.00001, amsgrad=False)           
                                                                        
        self.triplet_model_worker.compile(loss='mean_absolute_error', optimizer=adam_opt)
        print (self.triplet_model_worker.summary())
                                                                        
                                                                        
    def fit_model(self, pathname='./models/'):                          
        if not os.path.exists(pathname):                                
            os.makedirs(pathname)                                       
        if not os.path.exists(pathname+'/weights'):                     
            os.makedirs(pathname+'/weights')                            
        if not os.path.exists(pathname+'/tb'):                          
            os.makedirs(pathname+'/tb')                                 
        filepath=pathname+"weights/{epoch:02d}.hdf5"                    
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        tensorboard = TensorBoard(log_dir=pathname+'/tb', write_graph=True, write_images=True)
        callbacks_list = [checkpoint, tensorboard]                      
                                                                        
        #Parameter                                                              
        params = {'dim': (224, 224), 'batch_size':128, 'n_classes':11, 'n_channels':3, 'shuffle':True}
                                                                        
        #Datasets                                                               
        partition = pickle.load(open('../../data/bam_2_partition_triplet.pkl', 'rb'))          
        labels = pickle.load(open('../../data/bam_2_labels_triplet.pkl', 'rb'))            
                                                                        
        #Generators
        training_generator = DataGenerator(partition['train'], labels, **params)

        self.triplet_model_worker.fit_generator(generator = training_generator,  epochs = 60, use_multiprocessing=True, workers = 10, callbacks = callbacks_list, verbose = 1)
                                                                        
                                                                        
if __name__ == "__main__":                                              
    m = cnNet(224, 224, 3, 11, 0.3)                                   
    m.create_model()                                                    
    #m.triplet_model_worker.load_weights('models_orig_classification/yo/weights/24.hdf5', by_name=True)
    exit()                                                                    
    m.fit_model('/scratch/models/yo/') 
