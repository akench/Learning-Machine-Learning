import tensorflow as tf
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
import tensorflow.contrib.slim as slim
from utils.data_utils import DataUtil
import datetime

logs_path = 'logs/'
# tensorboard --logdir=run1:logs/ --port 6006
BATCH_SIZE = 128
Z_DIM = 100

X = tf.placeholder(dtype=tf.float32, shape=[None, 32*32])
Z = tf.placeholder(dtype=tf.float32, shape=[None, Z_DIM])




def sample_Z(m, n):
    # return np.random.uniform((size=m,n))
    return np.random.normal(size=(m, n))


def generator(Z):

    with tf.variable_scope('gen', reuse=tf.AUTO_REUSE):

        with slim.arg_scope([slim.fully_connected],
            weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):

            print('generator shapes')
            net = slim.fully_connected(Z, 2*2*256, activation_fn=None, scope='fc1')
            net = tf.reshape(net, (-1, 2, 2, 256))
            net = slim.batch_norm(net)
            net = tf.nn.leaky_relu(net)
            # shape = 2 x 2 x 256
            print(net.shape)

            net = slim.conv2d_transpose(net, 64, [5,5], stride=2, scope='convT1')
            net = slim.batch_norm(net)
            net = tf.nn.leaky_relu(net)
            # shape = 7 x 7 x 64
            print(net.shape)

            net = slim.conv2d_transpose(net, 16, [3,3], stride=2, scope='convT2')
            net = slim.batch_norm(net)
            net = tf.nn.leaky_relu(net)
            # shape = 15 x 15 x 16
            print(net.shape)

            net = slim.conv2d_transpose(net, 1, [4,4], stride=4, scope='convT3')
            net = slim.batch_norm(net)
            net = tf.nn.leaky_relu(net)
            # shape = 32 x 32 x 1
            print(net.shape)

            net = tf.nn.tanh(net)

    return net



def discriminator(X):

    with tf.variable_scope('disc', reuse=tf.AUTO_REUSE):

        net = tf.reshape(X, [-1, 32, 32, 1])

        with slim.arg_scope([slim.conv2d], padding='SAME', stride=2,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):

            with slim.arg_scope([slim.fully_connected],
                weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):

                print('discriminator shapes')

                net = slim.conv2d(net, 64, [5,5], scope='conv1')
                print(net.shape)
                # net = slim.batch_norm(net)
                net = tf.nn.leaky_relu(net)
                net = slim.avg_pool2d(net, [2,2], stride=2, scope='pool1')
                print(net.shape)

                net = slim.conv2d(net, 128, [5,5], scope='conv2')
                print(net.shape)
                # net = slim.batch_norm(net)
                net = tf.nn.leaky_relu(net)
                net = slim.avg_pool2d(net, [2,2], stride=2, scope='pool2')
                print(net.shape)

                net = slim.conv2d(net, 256, [4,4], scope='conv3')
                print(net.shape)
                # net = slim.batch_norm(net)
                net = tf.nn.leaky_relu(net)
                net = slim.avg_pool2d(net, [2,2], stride=2, scope='pool3')
                print(net.shape)

                net = slim.conv2d(net, 512, [3,3], scope='conv4')
                print(net.shape)
                # net = slim.batch_norm(net)
                net = tf.nn.leaky_relu(net)
                net = slim.avg_pool2d(net, [2,2], stride=2, scope='pool4')
                print(net.shape)

                net = slim.flatten(net, scope='flatten5')
                print(net.shape)
                net = slim.fully_connected(net, 128, activation_fn=tf.nn.relu, scope='fc6')
                net = slim.fully_connected(net, 1, activation_fn=None, scope='fc7')
                print(net.shape)

    return net


def train(continue_training=False):

    t0 = time.time()

    generated = generator(Z)
    d_logit_real = discriminator(X)
    d_logit_fake = discriminator(generated)

    with tf.name_scope('d_loss_real'):
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_real,
            labels=tf.random_uniform([BATCH_SIZE, 1], minval=0.7, maxval=1.3)
            # labels=tf.ones_like(d_logit_real)
            # labels=tf.fill([BATCH_SIZE, 1], 0.9)
        ))

    with tf.name_scope('d_loss_fake'):
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_fake,
            labels=tf.random_uniform([BATCH_SIZE, 1], minval=0.0, maxval=0.3)
            # labels=tf.zeros_like(d_logit_fake)
        ))

    with tf.name_scope('d_loss'):
        D_loss = D_loss_fake + D_loss_real

    with tf.name_scope('g_loss'):
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_fake,
            labels=tf.random_uniform([BATCH_SIZE, 1], minval=0.7, maxval=1.3)
            # labels=tf.ones_like(d_logit_fake)
        ))


    tf.summary.scalar("g_loss", G_loss)
    tf.summary.scalar('d_loss', D_loss)
    tf.summary.scalar('d_loss_fake', D_loss_fake)
    tf.summary.scalar('d_loss_real', D_loss_real)

    summary_op = tf.summary.merge_all()

    tvars = tf.trainable_variables()
    d_vars = [v for v in tvars if 'disc' in v.name]
    g_vars = [v for v in tvars if 'gen' in v.name]



    D_train_step = tf.train.AdamOptimizer(0.00001).minimize(D_loss, var_list=d_vars)
    G_train_step = tf.train.AdamOptimizer(.001).minimize(G_loss, var_list=g_vars)


    data_util = DataUtil(data_dir='data', batch_size=BATCH_SIZE,
            num_epochs=200, supervised=False)


    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        writer = tf.summary.FileWriter(logs_path, sess.graph)
        tf.train.write_graph(sess.graph_def, 'model', 'devanagari.pbtxt', True)

        if continue_training:
            saver.restore(sess, tf.train.latest_checkpoint('model/model.ckpt'))

        G_loss_curr = 1.0
        D_loss_curr = 1.0

        for it in range(1000000):

            if it % 1000 == 0:
                path = saver.save(sess, 'model/model.ckpt', global_step = it)
                print('path saved in %s' % (path))

            if it % 100 == 0:
                seed = np.full((1, 100), 0.0)
                sample = sess.run(generated, feed_dict={Z: seed})
                plt.axis('off')
                plt.imshow(sample.reshape(32, 32), cmap='Greys_r')
                plt.savefig('out/{}.png'.format(it), bbox_inches='tight')
                plt.clf()
                plt.cla()
                plt.close()

            batch_x  = data_util.get_next_batch()

            if batch_x is None:
                path = saver.save(sess, 'model/model.ckpt', global_step = it)
                print('path saved in %s' % (path))
                return


            if it % 1 == 0:
                _, G_loss_curr, G_summary = sess.run([G_train_step, G_loss, summary_op],
                    feed_dict={Z: sample_Z(BATCH_SIZE, Z_DIM), X: batch_x})

            if it % 2 == 0:
                _, D_loss_curr, D_summary = sess.run([D_train_step, D_loss, summary_op],
                    feed_dict={Z: sample_Z(BATCH_SIZE, Z_DIM), X: batch_x})

            writer.add_summary(G_summary, global_step=data_util.global_num)
            writer.add_summary(D_summary, global_step=data_util.global_num)

            if it % 10 == 0:
                print('Iter: {}'.format(it))
                print('D_loss: {:.4}'.format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))
                print()

    print(str(datetime.timedelta(seconds=time.time() - t0)))


def gen_images(num):

    generated = generator(Z)

    with tf.Session() as sess:
        saver=tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('model/'))

        images = sess.run(generated, feed_dict={Z: sample_Z(num, 100)})

        for i, img in enumerate(images):

            plt.axis('off')
            plt.imshow(img.reshape(32, 32), cmap='Greys_r')
            plt.savefig('gen_images/{}.png'.format(i), bbox_inches='tight')
            plt.close()



# gen_images(50)
train()
