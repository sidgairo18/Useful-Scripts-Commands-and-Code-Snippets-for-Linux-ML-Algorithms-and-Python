#!/usr/bin/python                                                       
# Author: Siddhartha Gairola (siddhartha dot gairola at iiit dot ac dot in)
import numpy as np
import cv2
import os

#This is a class to find Precision, Recall, Average Precision and Mean Average Precsion
'''Please note - this is a very unoptimized version of these funtions : I had to get this done fast and did not want to get into reading documentation
for available libraries like scikit-learn. You may refer to those for more efficient implementations of the same'''
#Expecting subset x : where each row is a data point; y_subset[i] : containts label corresponding to ith entry in subest x 
#Expecting query set X : where each row is a data point; y_query[i] : contains the label corresponding to the ith entry in query X

class mAP():

    def __init__(self, X, x, y_subset, y_query, k):

        self.X = X
        self.x = x
        self.y_subset = y_subset
        self.y_query = y_query
        self.k = k
        self.knns = None
        self.precision_array = None

    def compute_distances(self):
        print ("Computing distances")
        num_test = self.x.shape[0]
        num_train = self.X.shape[0]
        dists = np.zeros((num_test, num_train))
        dists = np.sqrt((self.x**2).sum(axis=1, keepdims=True) + (self.X**2).sum(axis=1) - 2*(self.x).dot((self.X).T))
        print (dists)

        return dists

    def get_knn(self, dists):
        knns = []
        for i in range(dists.shape[0]):
            l = list(np.argsort(dists[i,:]))[:self.k+1]
            knns.append(l[1:])
        knns = np.asarray(knns)
        self.knns = knns

    def precision(self, k):

        precision_array = np.zeros(self.x.shape[0])

        for i in range(self.x.shape[0]):

            #for ith query
            cur_label = self.y_subset[i]
            relevant_imgs = 0.0
            retrieved_imgs = float(self.k)

            for j in range(k):
                
                #retrieved index
                ret_idx = self.knns[i,j]
                ret_label = self.y_query[ret_idx]
                if ret_label == cur_label:
                    relevant_imgs += 1.0

            precision_array[i] = relevant_imgs/retrieved_imgs

        return precision_array
    
    def recall(self, k):

        recall_array = np.zeros(self.x.shape[0])

        for i in range(self.x.shape[0]):

            #for ith query
            cur_label = self.y_subset[i]
            relevant_imgs = 0.0
            total_relevant_imgs = 0.0

            for j in range(k):
                
                #retrieved index
                ret_idx = self.knns[i,j]
                ret_label = self.y_query[ret_idx]
                if ret_label == cur_label:
                    relevant_imgs += 1.0

            #calculating total number of relevant imgs in the retrieval set

            for j in range(self.y_query.shape[0]):
                if self.y_query[j] == cur_label:
                    total_relevant_imgs += 1

            recall_array[i] = relevant_imgs/total_relevant_imgs

        return recall_array

    def average_precision(self):

        no_of_retrieved_docs = self.k
        Precision_array = np.zeros((self.x.shape[0], self.k))

        for i in range(self.k):
            print ("Average Precision Index", i)
            Precision_array[:,i] = self.precision(i+1)

        self.precision_array = Precision_array

        average_precision_array = np.zeros(self.x.shape[0])

        for i in range(self.x.shape[0]):
            
            #for the ith query
            cur_label = self.y_subset[i]
            total_relevant_imgs = 0.0
            cur_average_precision = 0.0

            for j in range(no_of_retrieved_docs):
                
                ret_idx = self.knns[i,j]
                ret_label = self.y_query[ret_idx]
                if ret_label == cur_label:
                    cur_average_precision += Precision_array[i,j]
                    #calculating total numnber of relevant images for the current candidate
                    total_relevant_imgs += 1.0


            cur_average_precision /= total_relevant_imgs

            average_precision_array[i] = cur_average_precision

        return average_precision_array

    def mean_average_precision(self):

        average_precision_array = self.average_precision()

        maP = 0.0

        for i in range(self.x.shape[0]):
            maP += average_precision_array[i]

        maP /= self.x.shape[0]
        return maP
