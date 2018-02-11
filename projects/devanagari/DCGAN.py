import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow.contrib.slim as slim


X = tf.placeholder(dtype=tf.float32, shape=[None, 32*32])
Z = tf.placeholder(dtype=tf.float32, shape=[None, 100])


def sample_z(m, n):
	return np.random.normal(size=(m, n))


def generator(Z):

	with tf.variable_scope('gen', reuse=tf.AUTO_REUSE):

		with slim.arg_scope([slim.fully_connected],
			weights_initializer=tf.contrib.layers.variance_scaling_initializer(uniform=False),
			weights_regularizer=slim.l2_regularizer(.05)):

			net = slim.fully_connected(Z, 264*2*2, activation_fn=None, scope='fc1')
			net = tf.reshape(net, (-1, 2, 2, 264))
			


def discriminator(X):

	with tf.variable_scope('disc', reuse=tf.AUTO_REUSE):

		net = tf.reshape(X, [-1, 32, 32, 1])

		with slim.arg_scope([slim.conv2d], padding='SAME', stride=2,
			weights_initializer=tf.contrib.layers.variance_scaling_initializer(uniform=False),
			weights_regularizer=slim.l2_regularizer(.05)):

			with slim.arg_scope([slim.fully_connected],
				weights_initializer=tf.contrib.layers.variance_scaling_initializer(uniform=False),
				weights_regularizer=slim.l2_regularizer(.05)):

				net = slim.conv2d(net, 64, [5,5], scope='conv1')
				net = slim.batch_norm(net)
				net = tf.nn.leaky_relu(net)
				net = slim.avg_pool2d(net, [2,2])

				net = slim.conv2d(net, 128, [5,5], scope='conv2')
				net = slim.batch_norm(net)
				net = tf.nn.leaky_relu(net)
				net = slim.avg_pool2d(net, [2,2])

				net = slim.conv2d(net, 256, [5,5], scope='conv3')
				net = slim.batch_norm(net)
				net = tf.nn.leaky_relu(net)
				net = slim.avg_pool2d(net, [2,2])

				net = slim.conv2d(net, 512, [5,5], scope='conv4')
				net = slim.batch_norm(net)
				net = tf.nn.leaky_relu(net)
				net = slim.avg_pool2d(net, [2,2])

				net = slim.flatten(net, scope='flatten5')
				net = slim.fully_connected(net, 128, activation_fn=tf.nn.relu)
				net = slim.fully_connected(net, 2, activation_fn=None)


	return net
