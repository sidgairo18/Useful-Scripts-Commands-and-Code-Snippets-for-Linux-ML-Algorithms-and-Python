import numpy as np
import cv2
from sklearn.datasets import fetch_mldata
import pandas as pd
import matplotlib.pyplot as plt
#matplotlit.use('Agg')
from sklearn.decomposition import PCA
import time
from sklearn.manifold import TSNE
from ggplot import *
print ("Import Successful")

train_file = 'bottle_neck_mnist.txt'
labels_file = 'mnist_test_labels.txt'


def data_loader():

    X_train = np.loadtxt(train_file)
    y_train = np.zeros(X_train.shape[0])
    
    d = {}
    f = open(labels_file)
    for i, line in enumerate(f):
        y_train[i] = float(line.strip())
    f.close()

   
    #get training data and test data
    print ("Training data shape after stacking", X_train.shape, y_train.shape)

    return X_train, y_train

X_train, y_train = data_loader()


X = X_train
#y = y_train[:,0] + 2*y_train[:,11]
y = y_train

print (X.shape, y.shape)

feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]


df = pd.DataFrame(X, columns=feat_cols)
df['label'] = y
df['label'] = df['label'].apply(lambda i: str(i))

X, y = None, None

#print 'Size of the dataframe: {}'.format(df.shape)

rndperm = np.random.permutation(df.shape[0])

'''
# Plot the graph
plt.gray()
fig = plt.figure( figsize=(16,7) )
for i in range(0,30):
    ax = fig.add_subplot(3,10,i+1, title='Digit: ' + str(df.loc[rndperm[i],'label']) )

ax.matshow(df.loc[rndperm[i],feat_cols].values.reshape((28,28)).astype(float))
plt.show()


pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)

df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1] 
df['pca-three'] = pca_result[:,2]

print 'Explained variation per principal component: {}'.format(pca.explained_variance_ratio_)

chart = ggplot( df.loc[rndperm[:3000],:], aes(x='pca-one', y='pca-two', color='label') ) + geom_point(size=75,alpha=0.8) + ggtitle("First and Second Principal Components colored by digit")

print chart

n_sne = 7000
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)

print 't-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start)

df_tsne = df.loc[rndperm[:n_sne],:].copy()
df_tsne['x-tsne'] = tsne_results[:,0]
df_tsne['y-tsne'] = tsne_results[:,1]

chart = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label') ) + geom_point(size=70,alpha=0.1) + ggtitle("tSNE dimensions colored by digit")
print chart
'''

pca_50 = PCA(n_components=100)
pca_result_50 = pca_50.fit_transform(df[feat_cols].values)

#print ('Cumulative explained variation for 50 principal components: ',(np.sum(pca_50.explained_variance_ratio_))

n_sne = 5000

time_start = time.time()

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_pca_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)
#tsne_pca_results = tsne.fit_transform(pca_result_50[rndperm[:n_sne]])


#print 't-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start)

df_tsne = None
df_tsne = df.loc[rndperm[:n_sne],:].copy()
df_tsne['x-tsne'] = tsne_pca_results[:,0]
df_tsne['y-tsne'] = tsne_pca_results[:,1]

chart = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label') ) + geom_point(size=70,alpha=0.8) + ggtitle("tSNE dimensions colored by Digit")
#print chart
chart.save('tsne_mnist_triplet.png')
