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

    with tf.variable_scope('gen', reuse=tf.AUTO_REUSE):

        G_w1 = tf.get_variable('g_w1', shape=[100, 128], initializer=tf.contrib.layers.xavier_initializer())
        G_b1 = tf.get_variable('g_b1', shape=[128], initializer=tf.contrib.layers.xavier_initializer())

        net = tf.matmul(z, G_w1) + G_b1
        net = tf.nn.relu(net)


        G_w2 = tf.get_variable('g_w2', shape=[128, 784], initializer=tf.contrib.layers.xavier_initializer())
        G_b2 = tf.get_variable('g_b2', shape=[784], initializer=tf.contrib.layers.xavier_initializer())


        net = tf.matmul(net, G_w2) + G_b2
        net = tf.nn.sigmoid(net)

    return net


def discriminator(x):

    with tf.variable_scope('dis', reuse=tf.AUTO_REUSE):

        D_w1 = tf.get_variable('d_w1', shape=[784, 128], initializer=tf.contrib.layers.xavier_initializer())
        D_b1 = tf.get_variable('d_b1', shape=[128], initializer=tf.contrib.layers.xavier_initializer())

        net = tf.matmul(x, D_w1) + D_b1
        net = tf.nn.relu(net)


        D_w2 = tf.get_variable('d_w2', shape=[128, 1], initializer=tf.contrib.layers.xavier_initializer())
        D_b2 = tf.get_variable('d_b2', shape=[1], initializer=tf.contrib.layers.xavier_initializer())

        net = tf.matmul(net, D_w2) + D_b2
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

    print(d_vars)
    print('')
    print('')
    print(g_vars)

    D_train_step = tf.train.AdamOptimizer().minimize(D_loss, var_list=d_vars)
    G_train_step = tf.train.AdamOptimizer().minimize(G_loss, var_list=g_vars)


    batch_size = 128
    Z_dim = 100


    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        g_saver = tf.train.Saver(var_list=g_vars)

        G_loss_curr = 1.0
        D_loss_curr = 1.0

        for it in range(100000):

            if it % 10000 == 0:
                path = g_saver.save(sess, 'generator_mnist/model.ckpt')

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


def gen_images(num_img):

    generated = generator(Z)

    with tf.Session() as sess:

        saver = tf.train.Saver()
        saver.restore(sess, 'generator_mnist/model.ckpt')

        images = sess.run(generated, feed_dict={Z: sample_Z(num_img, 100)})

        for i, img in enumerate(images):
            plt.axis('off')
            plt.imshow(img.reshape(28, 28), cmap='Greys_r')
            plt.savefig('gen_images/{}.png'.format(i), bbox_inches='tight')
            plt.close()

def gen_image_with_seed(seed):
    generated = generator(Z)

    with tf.Session() as sess:

        saver = tf.train.Saver()
        saver.restore(sess, 'generator_mnist/model.ckpt')

        image = sess.run(generated, feed_dict={Z: np.full([1, 100], seed)})

        plt.axis('off')
        plt.imshow(image.reshape(28, 28), cmap='Greys_r')
        plt.savefig('seed_gen/{}.png'.format(int(seed * 100 + 100)), bbox_inches='tight')
        plt.close()


# train()
# gen_images(50)

i = -100.0
while(i <= 100.0):
    gen_image_with_seed(i / 100)
    i += 1.0