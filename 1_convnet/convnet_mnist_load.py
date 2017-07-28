from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import tensorflow as tf

import numpy as np
import scipy.ndimage

network = input_data(shape=[None, 28, 28, 1], name='input')
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='/notebooks/logs/tflearn_logs/')

# Load a model
model.load('my_model.tflearn')

import sys


data = scipy.ndimage.imread(sys.argv[1], flatten=True)
#data = scipy.ndimage.imread("../testing5.png", flatten=True)
data = np.vectorize(lambda x: 255 - x)(np.reshape(data, (1,28,28,1)))
data = data/255.  # image value is 0.0 ~ 1.0

#predictions = np.array(model.predict(testX)).argmax(axis=1)
#actual = testY.argmax(axis=1)
#test_accuracy = np.mean(predictions == actual, axis=0)
#print("Test accuracy: ", test_accuracy)

#predictions = np.array(model.predict(data)).argmax(axis=1)
#print(predictions)       
result = model.predict(data)
print(result)
print('Predicted number = ' + str(np.argmax(result, 1)[0]))



