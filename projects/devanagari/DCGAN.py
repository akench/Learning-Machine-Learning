import tensorflow as tf
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import time
import tensorflow.contrib.slim as slim
from utils.data_utils import DataUtil
import datetime
import shutil
import os

logs_path = 'logs/'
# tensorboard --logdir=run1:logs/ --port 6006
BATCH_SIZE = 64
Z_DIM = 200

X = tf.placeholder(dtype=tf.float32, shape=[None, 28*28])
Z = tf.placeholder(dtype=tf.float32, shape=[None, Z_DIM])
keep_prob_placeholder = tf.placeholder(dtype=tf.float32, shape=())




def sample_Z(m, n):
    # return np.random.uniform((size=m,n))
    return np.random.normal(size=(m, n))


def generator(Z, keep_prob):

    with tf.variable_scope('gen', reuse=tf.AUTO_REUSE):

        with slim.arg_scope([slim.fully_connected],
            weights_initializer=tf.contrib.layers.xavier_initializer()):
             # weights_regularizer=slim.l2_regularizer(.05)):

            x = tf.layers.dense(Z, units=6 * 6 * 128)
            x = tf.nn.leaky_relu(x)
            x = tf.reshape(x, shape=[-1, 6, 6, 128])
            x = tf.layers.conv2d_transpose(x, 64, 4, strides=2)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.conv2d_transpose(x, 1, 2, strides=2)
            x = tf.nn.leaky_relu(x)
            x = tf.nn.tanh(x)
        return x
'''
             net = slim.fully_connected(Z, 6*6*128, activation_fn=None, scope='fc1')
             net = slim.dropout(net, keep_prob=keep_prob)
             net = slim.batch_norm(net)
             net = tf.nn.tanh(net)

             print(net.shape)

             net = tf.reshape(net, [-1,6,6,128])
             net = tf.layers.conv2d_transpose(net, 64, 4, strides=2)
             net = slim.batch_norm(net)
             net = tf.nn.tanh(net)
             print(net.shape)

             net = tf.layers.conv2d_transpose(net, 1, 2, strides=2)
             print(net.shape)

             net = tf.nn.tanh(net)
'''
   # return x



def discriminator(X):

    with tf.variable_scope('disc', reuse=tf.AUTO_REUSE):

        net = tf.reshape(X, [-1, 28, 28, 1])

        tf.summary.image('input to disc', net, 4)

        with slim.arg_scope([slim.conv2d], padding='SAME', stride=2,
            weights_initializer=tf.contrib.layers.xavier_initializer()):
             # weights_regularizer=slim.l2_regularizer(.05)):

            with slim.arg_scope([slim.fully_connected],
                weights_initializer=tf.contrib.layers.xavier_initializer()):
                 # weights_regularizer=slim.l2_regularizer(.05)):

                print('discriminator shapes')

                x = tf.layers.conv2d(net, 64, 5)
                x = tf.nn.tanh(x)
                x = tf.layers.average_pooling2d(x, 2, 2)
                x = tf.layers.conv2d(x, 128, 5)
                x = tf.nn.tanh(x)
                x = tf.layers.average_pooling2d(x, 2, 2)
                x = tf.contrib.layers.flatten(x)
                x = tf.layers.dense(x, 1024)
                x = tf.nn.tanh(x)
                x = tf.layers.dense(x, 1)

               # net = slim.conv2d(net, 64, [5,5], scope='conv1')
                #print(net.shape)
                # net = slim.batch_norm(net)
               # net = tf.nn.tanh(net)
               # net = slim.avg_pool2d(net, [2,2], stride=2, scope='pool1')
               # print(net.shape)

               # net = slim.conv2d(net, 128, [5,5], scope='conv2')
               # print(net.shape)
               # # net = slim.batch_norm(net)
               # net = tf.nn.tanh(net)
               # net = slim.avg_pool2d(net, [2,2], stride=2, scope='pool2')
               # print(net.shape)


               # net = slim.flatten(net, scope='flatten5')
               # print(net.shape)
               # net = slim.fully_connected(net, 1024, activation_fn=tf.nn.tanh, scope='fc6')
               # net = slim.fully_connected(net, 1, activation_fn=None, scope='fc7')
               # print(net.shape)

    return x


def plot_samples(samples):

    fig = plt.figure()
    gs = gridspec.GridSpec(3,3)
    seed_i = 0

    for r in range(3):
        for c in range(3):
            ax = plt.subplot(gs[r, c])
            ax.axis('off')
            ax.imshow(samples[seed_i].reshape(28, 28), cmap='Greys_r')
            seed_i += 1

    return fig


def train(continue_training=False):

    shutil.rmtree('logs')
    os.mkdir('logs')

    t0 = time.time()

    generated = generator(Z, keep_prob_placeholder)
    d_logit_real = discriminator(X)
    d_logit_fake = discriminator(generated)


    with tf.name_scope('d_loss_real'):
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_real,
            labels=tf.random_uniform([BATCH_SIZE, 1], minval=.9, maxval=1.1)
            # labels=tf.random_normal([BATCH_SIZE, 1], mean=1.0, stddev=0.05)
        ))

    with tf.name_scope('d_loss_fake'):
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_fake,
            # labels = tf.random_normal([BATCH_SIZE, 1], mean=0.0, stddev=0.05)
            labels=tf.random_uniform([BATCH_SIZE, 1], minval=0.0, maxval=0.2)
        ))

    with tf.name_scope('d_loss'):
        D_loss = D_loss_fake + D_loss_real

    with tf.name_scope('g_loss'):
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_fake,
            # labels=tf.random_normal([BATCH_SIZE, 1], mean=1.0, stddev=0.05)
            labels=tf.random_uniform([BATCH_SIZE, 1], minval=.9, maxval=1.1)
        ))


    tf.summary.scalar("1 Generator Loss", G_loss)
    tf.summary.scalar('2 Discriminator Loss', D_loss)
    tf.summary.scalar('d_loss_fake', D_loss_fake)
    tf.summary.scalar('d_loss_real', D_loss_real)

    summary_op = tf.summary.merge_all()

    tvars = tf.trainable_variables()
    d_vars = [v for v in tvars if 'disc' in v.name]
    g_vars = [v for v in tvars if 'gen' in v.name]



    D_train_step = tf.train.AdamOptimizer(0.001).minimize(D_loss, var_list=d_vars)
    G_train_step = tf.train.AdamOptimizer(0.0001).minimize(G_loss, var_list=g_vars)


    data_util = DataUtil(data_dir='data', batch_size=BATCH_SIZE,
            num_epochs=300, supervised=False)


    seeds = sample_Z(9, Z_DIM)


    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        writer = tf.summary.FileWriter(logs_path, sess.graph)
        tf.train.write_graph(sess.graph_def, 'model', 'devanagari.pbtxt', True)

        if continue_training:
            saver.restore(sess, tf.train.latest_checkpoint('model/'))
        else:
            shutil.rmtree('model/')
            os.mkdir('model/')
            shutil.rmtree('out/')
            os.mkdir('out/')

        G_loss_curr = 1.0
        D_loss_curr = 1.0


        for it in range(1000000):

            #save model
            if it % 1000000000 == 0:
                path = saver.save(sess, 'model/model.ckpt', global_step = it)
                print('path saved in %s' % (path))


            #plot samples
            if it % 500 == 0:
                samples = sess.run(generated, feed_dict={Z: seeds, keep_prob_placeholder:1.0})
                fig = plot_samples(samples)
                fig.savefig('out/{}.png'.format(it), bbox_inches='tight')



            batch_x  = data_util.get_next_batch()

            if batch_x is None:
                path = saver.save(sess, 'model/model.ckpt', global_step = it)
                print('path saved in %s' % (path))
                return


            if it % 1 == 0:
                _, G_loss_curr, G_summary = sess.run([G_train_step, G_loss, summary_op],
                    feed_dict={Z: sample_Z(BATCH_SIZE, Z_DIM), X: batch_x, keep_prob_placeholder: .5})

            if it % 2 == 0:
                _, D_loss_curr, D_summary = sess.run([D_train_step, D_loss, summary_op],
                    feed_dict={Z: sample_Z(BATCH_SIZE, Z_DIM), X: batch_x, keep_prob_placeholder: .5})




            if it % 10 == 0:
                writer.add_summary(G_summary, global_step=data_util.global_num)
                writer.add_summary(D_summary, global_step=data_util.global_num)
                print('Iter: {}'.format(it))
                print('D_loss: {:.4}'.format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))
                print()

    print(str(datetime.timedelta(seconds=time.time() - t0)))


def gen_images(num):

    generated = generator(Z, keep_prob_placeholder)

    with tf.Session() as sess:
        saver=tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('model/'))

        images = sess.run(generated, feed_dict={Z: sample_Z(num, 200), keep_prob_placeholder:1.0})

        for i, img in enumerate(images):

            plt.axis('off')
            plt.imshow(img.reshape(28, 28), cmap='Greys_r')
            plt.savefig('gen_images/{}.png'.format(i), bbox_inches='tight')
            plt.close()
        print('finished generating!')



#gen_images(50)
train(continue_training=False)
