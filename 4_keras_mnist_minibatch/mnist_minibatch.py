'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import sys

import numpy as np

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

def genBatch_withPreprop_train(batch_size, path='mnist.npz'):
    ## train_or_test: 'train', 'test'
    path = keras.utils.get_file(path,
                    origin='https://s3.amazonaws.com/img-datasets/mnist.npz',
                    file_hash='8a61469f7ea1b51cbae51d4f78837e45')
    f = np.load(path)
    x_train_ori, y_train_ori = f['x_train'], f['y_train']
    #x_test_ori, y_test_ori = f['x_test'], f['y_test']
    f.close()
    
    sample_per_batch = x_train_ori.shape[0] // batch_size
    
    start = 0
    end = batch_size
    endFlag = 0
    while 1:
        x_train_List = []
        y_train_List = []
        for i in range(start, end):
            x_train = x_train_ori[i, :, :]
            y_train = y_train_ori[i]


            x_train = x_train.reshape( img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

            x_train = x_train.astype('float32')
            x_train /= 255
            x_train_List.append(x_train)
            #x_train = np.expand_dims(x_train, axis=0)

            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_train_List.append(y_train)
            #y_train = np.expand_dims(y_train, axis=0)
        
        out_x_train = np.array(x_train_List)    
        out_y_train = np.array(y_train_List)    
        yield (out_x_train, out_y_train)
        if endFlag == 1:
            start = 0
            end = batch_size
            endFlag = 0
        else:
            start += batch_size
            end += batch_size
        if end > x_train_ori.shape[0]:
            end = x_train_ori.shape[0]
            endFlag = 1

        
        
(_, _), (x_test, y_test) = mnist.load_data()

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_test = x_test.astype('float32')
x_test /= 255
y_test = keras.utils.to_categorical(y_test, num_classes)



            
# the data, shuffled and split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()



model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test))

sampleNum = 60000
steps_per_epo = sampleNum // batch_size
if sampleNum % batch_size != 0:
    steps_per_epo += 1

model.fit_generator(generator = genBatch_withPreprop_train(batch_size),
                    steps_per_epoch = steps_per_epo,
                    epochs = 12,
                    verbose=1)
                    #validation_data = genBatch_withPreprop_test,
                    #validation_steps = 10000 //batch_size,
                    #verbose=1)
                    
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
