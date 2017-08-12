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
from keras.models import load_model

import os
import numpy as np
from scipy import misc

def train():
    batch_size = 128
    num_classes = 48
    epochs = 50

    # input image dimensions
    img_rows, img_cols = 28, 28


    # each catogries: train 360, test remaining

    # the data, shuffled and split between train and test sets
    num_train = 360
    (x_train, y_train), (x_test, y_test) = _get_filenames_and_classes('/notebooks/mnist_prac/3_convet_keras', num_train)
    
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    #https://keras.io/datasets/
    #x_train, x_test: uint8 array of grayscale image data with shape (num_samples, 28, 28).
    #y_train, y_test: uint8 array of digit labels (integers in range 0-9) with shape (num_samples,).

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    #print(y_test.shape[0])
    #print(np.arange(y_test.shape[0]))
    #y = np.array(y, dtype='int').ravel()
    #return
    y_test = keras.utils.to_categorical(y_test, num_classes)

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

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))


    model.save('my_model.h5')

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    #Test accuracy: 0.9901

def _get_filenames_and_classes(dataset_dir, num_train):
    """Returns a list of filenames and inferred class names.
    Args:
      dataset_dir: A directory containing a set of subdirectories representing
        class names. Each subdirectory should contain PNG or JPG encoded images.
      num_train: number of training data
    Returns:
      A list of image file paths, relative to `dataset_dir` and the list of
      subdirectories, representing class names.
    """
    # print 'DATASET DIR:', dataset_dir
    # print 'subdir:', [name for name in os.listdir(dataset_dir)]
    # dataset_main_folder_list = []
    # for name in os.listdir(dataset_dir):
    # 	if os.path.isdir(name):
    # 		dataset_main_folder_list.append(name)
    dataset_main_folder_list = [name for name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir,name))]
    dataset_root = os.path.join(dataset_dir, dataset_main_folder_list[0])
    directories = []
    class_names = []
    for filename in os.listdir(dataset_root):
        path = os.path.join(dataset_root, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)
   
    x_train = []
    x_test = []
    y_train = []
    y_test = []
        
    label_count = 0
    for directory in directories:
        count = 0
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            image = misc.imread(path)
            if len(image) != 28:
                print('error')
                print(len(image))
            if count < num_train:
                x_train.append(image)
                y_train.append(label_count)
            else:
                x_test.append(image)
                y_test.append(label_count)
            count = count + 1
        label_count = label_count + 1
            

    #print(x_train[1])            
    print('x_train shape:', np.array(x_train).shape)        
    print('x_test shape:', np.array(x_test).shape)  
    print('y_train shape:', np.array(y_train).shape)  
    print('y_test shape:', np.array(y_test).shape)

    print('label:', label_count)
    #x_train, x_test: uint8 array of grayscale image data with shape (num_samples, 28, 28).
    #y_train, y_test: uint8 array of digit labels (integers in range 0-9) with shape (num_samples,).
    #return photo_filenames, sorted(class_names)
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    return (x_train, y_train), (x_test, y_test)
    
def main():
    train()
    
    
    
if __name__ == "__main__":
    main()
    
    