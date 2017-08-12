from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import tensorflow as tf

import numpy as np
import scipy.ndimage

import sys

## Define network
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

## Build model
model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='/notebooks/logs/tflearn_logs/')

## Load a model from file
model.load('my_model.tflearn')

## Load testing image from command line argument
##  example:
##          python3 convnet_mnist_load.py ../test.png
data = scipy.ndimage.imread(sys.argv[1], flatten=True)
data = np.vectorize(lambda x: 255 - x)(np.reshape(data, (1,28,28,1)))
data = data/255.  # image value is 0.0 ~ 1.0


result = model.predict(data)
print(result)
print('Predicted number = ' + str(np.argmax(result, 1)[0]))

## Predict on mnist and print accuracy
#predictions = np.array(model.predict(testX)).argmax(axis=1)
#actual = testY.argmax(axis=1)
#test_accuracy = np.mean(predictions == actual, axis=0)
#print("Test accuracy: ", test_accuracy)
