import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/mnist_data', one_hot=True)

X = tf.placeholder(dtype=tf.float32, shape=[None, 784])
Z = tf.placeholder(dtype=tf.float32, shape=[None, 100])

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):

    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.fully_connected], weights_initializer=
            tf.truncated_normal_initializer(stddev=.02), weights_regularizer=
            slim.l2_regularizer(.05)):

            net = slim.fully_connected(z, 128, scope='g_fc1')
            net = tf.nn.relu(net)
            net = slim.fully_connected(net, 256, scope='g_fc2')
            net = tf.nn.relu(net)
            net = slim.fully_connected(net, 512, scope='g_fc3')
            net = tf.nn.relu(net)
            net = slim.fully_connected(net, 784, scope='g_fc4')
            net = tf.nn.sigmoid(net)
    return net


def discriminator(x):

    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.fully_connected], weights_initializer=
            tf.truncated_normal_initializer(stddev=.02), weights_regularizer=
            slim.l2_regularizer(.05)):

            net = slim.fully_connected(x, 512, scope='d_fc1')
            net = tf.nn.relu(net)
            net = slim.fully_connected(net, 256, scope='d_fc2')
            net = tf.nn.relu(net)
            net = slim.fully_connected(net, 128, scope='d_fc3')
            net = tf.nn.relu(net)
            net = slim.fully_connected(net, 1, scope='d_fc4')
            prob = tf.nn.sigmoid(net)

    return prob, net


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

        for it in range(100000):

            if it % 100 == 1:
                sample = sess.run(generated, feed_dict={Z: sample_Z(1, Z_dim)})
                plt.axis('off')
                plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
                plt.savefig('out/{}.png'.format(it), bbox_inches='tight')
                plt.clf()
                plt.cla()
                plt.close()

            batch_x, _ = mnist.train.next_batch(batch_size)


            _, D_loss_curr = sess.run([D_train_step, D_loss], 
                feed_dict={X: batch_x, Z: sample_Z(batch_size, Z_dim)})

            _, G_loss_curr = sess.run([G_train_step, G_loss],
                feed_dict={Z: sample_Z(batch_size, Z_dim)})

            if it % 100 == 0:
                print('Iter: {}'.format(it))
                print('D loss: {:.4}'. format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))
                print()


train()