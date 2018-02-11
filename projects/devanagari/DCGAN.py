import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow.contrib.slim as slim
from utils.data_utils import DataUtil


BATCH_SIZE = 128
Z_DIM = 100

X = tf.placeholder(dtype=tf.float32, shape=[None, 32*32])
Z = tf.placeholder(dtype=tf.float32, shape=[None, Z_DIM])




def sample_Z(m, n):
	return np.random.normal(size=(m, n))


def generator(Z):

	with tf.variable_scope('gen', reuse=tf.AUTO_REUSE):

		with slim.arg_scope([slim.fully_connected],
			weights_initializer=tf.contrib.layers.variance_scaling_initializer(uniform=False),
			weights_regularizer=slim.l2_regularizer(.05)):

			net = slim.fully_connected(Z, 2*2*256, activation_fn=None, scope='fc1')
			net = tf.reshape(net, (-1, 2, 2, 256))
			net = slim.batch_norm(net)
			net = tf.nn.leaky_relu(net)
			# shape = 2 x 2 x 256

			net = slim.conv2d_transpose(net, 64, [5,5], strides=2, scope='convT1')
			net = slim.batch_norm(net)
			net = tf.nn.leaky_relu(net)
			# shape = 7 x 7 x 64

			net = slim.conv2d_transpose(net, 16, [3,3], scope='convT2')
			net = slim.batch_norm(net)
			net = tf.nn.leaky_relu(net)
			# shape = 15 x 15 x 16

			net = slim.conv2d_transpose(net, 1, [4,4], scope='convT3')
			net = slim.batch_norm(net)
			net = tf.nn.leaky_relu(net)
			# shape = 32 x 32 x 1

			net = tf.nn.sigmoid(net)

	return net



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


def train(continue_training=False):

	generated = generator(Z)
	d_logit_real = discriminator(X)
	d_logit_fake = discriminator(generated)

	D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
		logits=d_logit_real,
		labels=tf.fill([BATCH_SIZE, 1], 0.9)
	))
	D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
		logits=d_logit_fake,
		labels=tf.zeros_like(d_logit_fake)
	))

	D_loss = D_loss_fake + D_loss_real

	G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
		logits=d_logit_fake,
		labels=tf.ones_like(d_logit_fake)
	))

	tvars = tf.trainable_variables()
	d_vars = [v for v in tvars if 'disc' in var.name]
	g_vars = [v for v in tvars if 'gen' in var.name]

	D_train_step = tf.train.AdamOptimizer(D_loss, var_list=d_vars)
	G_train_step = tf.train.AdamOptimizer(G_loss, var_list=g_vars)

	data_util = DataUtil(data_dir='data', batch_size=BATCH_SIZE,
			num_epochs=5, GAN = True)


	with tf.Session() as sess:

		sess.run(tf.global_variables_initializer())

		saver = tf.train.Saver()

		if continue_training:
			saver.restore(sess, tf.train.latest_checkpoint('model/model.ckpt'))

		G_loss_curr = 1.0
		D_loss_curr = 1.0

		for it in range(1000000):

			if it % 10000 == 0:
				path = saver.save(sess, 'model/model.ckpt', global_step = it)
				print('path saved in %s' % (path))

			if it % 1000 == 0:
				seed = np.full((1, 100), 0.0)
				sample = sess.run(generated, feed_dict={Z: seed})
				plt.axis('off')
				plt.imshow(sample.reshape(32, 32), cmap='Greys_r')
				plt.save_fig('out/{}.jpg'.format(it), bbox_inches='tight')
				plt.clf()
				plt.cla()
				plt.close()

			batch_x  = data_util.get_next_batch()

			if batch_x is None:
				path = saver.save(sess, 'model/model.ckpt', global_step = it)
				print('path saved in %s' % (path))
				return


			_, G_loss_curr = sess.run([G_train_step, G_loss],
					feed_dict={Z: sample_Z(BATCH_SIZE, Z_DIM)})

			_, D_loss_curr = sess.run([D_train_step, D_loss],
					feed_dict={Z: sample_Z(BATCH_SIZE, Z_DIM), X: batch_x})

			if it % 100 == 0:
                print('Iter: {}'.format(it))
                print('D_loss: {:.4}'.format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))
                print()
