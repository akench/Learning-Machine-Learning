import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import time

mnist = input_data.read_data_sets('/tmp/mnist_data', one_hot=True)



def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


X = tf.placeholder(dtype=tf.float32, shape=[None, 784])
Z = tf.placeholder(dtype=tf.float32, shape=[None, 100])




def sample_Z(m, n):
    # return np.full([m, n], 0.0)
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):

    G_W1 = tf.Variable(xavier_init([100, 128]), name='g_w1')
    G_b1 = tf.Variable(tf.zeros(shape=[128]), name='g_b1')

    net = tf.matmul(z, G_W1) + G_b1
    net = tf.nn.relu(net)

    G_W2 = tf.Variable(xavier_init([128, 784]), name='g_w2')
    G_b2 = tf.Variable(tf.zeros(shape=[784]), name='g_b2')


    net = tf.matmul(net, G_W2) + G_b2
    net = tf.nn.sigmoid(net)

    return net


def discriminator(x):

    D_W1 = tf.Variable(xavier_init([784, 1]), name='d_w1')
    D_b1 = tf.Variable(tf.zeros(shape=[1]), name='d_b1')

    net = tf.matmul(x, D_W1) + D_b1
    net = tf.nn.relu(net)


    D_W2 = tf.Variable(xavier_init([128, 1]), name='d_w2')
    D_b2 = tf.Variable(tf.zeros(shape=[1]), name='d_b2')

    net = tf.matmul(net, D_W2) + D_b2
    return net


def train():

    generated = generator(Z)
    d_logit_real = discriminator(X)
    d_logit_fake = discriminator(generated)

    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_logit_real,
        labels=tf.ones_like(d_logit_real)))

    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_logit_fake,
        labels=tf.zeros_like(d_logit_fake)))

    D_loss = D_loss_real + D_loss_fake


    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_logit_fake,
        labels=tf.ones_like(d_logit_fake)))


    tvars = tf.trainable_variables()
    d_vars = [var for var in tvars if 'd_' in var.name]
    g_vars = [var for var in tvars if 'g_' in var.name]

    D_train_step = tf.train.AdamOptimizer().minimize(D_loss, var_list=d_vars)
    G_train_step = tf.train.AdamOptimizer().minimize(G_loss, var_list=g_vars)


    batch_size = 128
    Z_dim = 100


    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        G_loss_curr = 1.0
        D_loss_curr = 1.0

        for it in range(100000):

            if it % 1000 == 0:
                sample = sess.run(generated, feed_dict={Z: sample_Z(1, Z_dim)})
                plt.axis('off')
                plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
                plt.savefig('out/{}.png'.format(it), bbox_inches='tight')
                plt.clf()
                plt.cla()
                plt.close()

            batch_x, _ = mnist.train.next_batch(batch_size)

            _, G_loss_curr = sess.run([G_train_step, G_loss],
                feed_dict={Z: sample_Z(batch_size, Z_dim)})
            _, D_loss_curr = sess.run([D_train_step, D_loss],
                feed_dict={X: batch_x, Z: sample_Z(batch_size, Z_dim)})


            if it % 100 == 0:
                print('Iter: {}'.format(it))
                print('D_loss: {:.4}'.format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))
                print()


train()