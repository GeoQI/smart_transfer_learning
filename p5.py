import numpy as np
import datetime
from scipy.misc import imsave
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
import tensorflow as tf
K._LEARNING_PHASE = tf.constant(0)
K.set_learning_phase(0)
nb_classes = 5
# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)


# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


X_train_1 = X_train[y_train == 1]
X_train_2 = X_train[y_train == 2]
X_train_3 = X_train[y_train == 3]
X_train_4 = X_train[y_train == 4]
X_train_5 = X_train[y_train == 5]
X_train_6 = X_train[y_train == 6]
X_train_7 = X_train[y_train == 7]
X_train_8 = X_train[y_train == 8]
X_train_9 = X_train[y_train == 9]
X_train_0 = X_train[y_train == 0]

count=100
v1=np.vstack((X_train_0[:count],X_train_1[:count]))
v1=np.vstack((v1,X_train_2[:count]))
v1=np.vstack((v1,X_train_3[:count]))
v1=np.vstack((v1,X_train_4[:count]))
v1=np.vstack((v1,X_train_5[:count]))
v1=np.vstack((v1,X_train_6[:count]))
v1=np.vstack((v1,X_train_7[:count]))
v1=np.vstack((v1,X_train_8[:count]))
v1=np.vstack((v1,X_train_9[:count]))

v2=np.zeros(count*10)
for x in range(count*10):
        v2[x]=x/count


model = load_model('mymodel.h5')
    
from keras import backend as K
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[3].output])
                                  
get_1rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[1].output])

get_7rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[7].output])
layer_output = get_3rd_layer_output([X])[0]
tt=get_3rd_layer_output([v1])[0]

t1 = t1.reshape((t1.shape[0], -1))



#visualize

import numpy as np
#from skdata.mnist.views import OfficialImageClassification
from matplotlib import pyplot as plt
from tsne import bh_sne

# load up data


# convert image data to float64 matrix. float64 is need for bh_sne
#x_data = np.asarray(x_data).astype('float64')
x_data = t4
x_data = np.asarray(x_data).astype('float64')
# For speed of computation, only run on a subset


# perform t-SNE embedding
vis_data = bh_sne(x_data)

# plot the result
vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]

plt.scatter(vis_x, vis_y, c=v2, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()











