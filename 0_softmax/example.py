import random

import sys, getopt

from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import scipy.ndimage



def training():
    ## Read data
    with tf.Session() as sess:
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        ## Regrassion model
        x = tf.placeholder(tf.float32, [None, 784], 'x')
        W = tf.Variable(tf.zeros([784, 10]), 'W')
        b = tf.Variable(tf.zeros([10]), 'b')

        y = tf.nn.softmax(tf.matmul(x, W) + b, name="y")
        y_ = tf.placeholder(tf.float32, [None, 10])

        ## Loss function
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        init = tf.global_variables_initializer()
        
        all_saver = tf.train.Saver()

        sess.run(init)

        for i in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(1000)
            sess.run(train_step, feed_dict= {x: batch_xs, y_: batch_ys})

        print ("done with training")
        all_saver.save(sess, './model_w/data.ckpt')
        
        


def testing():
    new_saver = tf.train.import_meta_graph('./model_w/data.ckpt.meta')
    graph = tf.get_default_graph()
    
    with tf.Session() as sess: 
        new_saver.restore(sess, './model_w/data.ckpt')
                
        data = np.vectorize(lambda x: 255 - x)(np.ndarray.flatten(scipy.ndimage.imread("../testing3.png", flatten=True)))

        
        op_to_restore = graph.get_tensor_by_name("y:0")
        print(sess.run(tf.argmax(op_to_restore,1), feed_dict = {"x:0" : [data]}))

    
def main(argv):
    try:
        opts, args = getopt.getopt(argv,"re",["training=","testing="])
    except getopt.GetoptError:
        print('example.py --training/--testing')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-r',"--training"):    
            training()
        elif opt in ("-e", "--esting"):
            testing()

if __name__ == "__main__":
    main(sys.argv[1:])