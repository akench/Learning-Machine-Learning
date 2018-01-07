from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data


import argparse
import sys
import tensorflow as tf
FLAGS = None

def main(_):
	#import MNIST data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	#put the images into this. The number of images goes into the none
	#section, since each image has 784 pixels (28x28)
    x = tf.placeholder(tf.float32, [None, 784])

	#creates an empty array of weights, we want to multiply the 784
	#image by it, to produce a 10 dimentional result of the evidence
	#bias is size 10, since we add each bias to each result of x*W

    W1 = tf.Variable(tf.zeros([784, 10]))
    b1 = tf.Variable(tf.zeros([10]))


	#multiplies the input by the weights, then adds a bias
	#uses the softmax function on it to get our probability distribution
	#vector
    y = tf.matmul(x, W1) + b1
    a= tf.nn.relu(y)
    # y = tf.matmul(a, W2) + b2
	#placeholder for the actual results
    y_ = tf.placeholder(tf.float32, [None, 10])

	#cross entropy method, measures how inefficient our method is
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    #launch the model in an interacive session
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    #with tf.Session() as sess:
    #run the training step 1000 times
    for _ in range(1000):
        #gets 100 random data points from the training set
        batch_xs, batch_ys = mnist.train.next_batch(100)
        #replaces the placeholders
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    #argmax function finds the max value in the tensor, basically finds the digit out program
    #things the data is
    #if the actual result equals our result, we get a true for correct_prediction
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

    #finds the accuracy of our predictions by averaging the true and false array
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
