import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from random import *
import time
from sklearn.utils import shuffle


data_placeholder = tf.placeholder(shape=[None, 2], dtype = tf.float32)
label_placeholder = tf.placeholder(shape=[None], dtype = tf.float32)

def model(data):

	data = tf.reshape(data, [-1, 2])
	with tf.variable_scope('regression_model'):
		with slim.arg_scope([slim.fully_connected], weights_initializer=tf.contrib.layers.variance_scaling_initializer(uniform = False)):
			data = slim.fully_connected(data, 1, activation_fn = None, scope = 'output')
	return data


def prepare_data(num):

	data = []
	labels = []
	for _ in range(num):
		x = randint(0, 9)
		y = randint(0, 9)
		z = x + y
		temp = []
		temp.append(x)
		temp.append(y)
		data.append(temp)
		labels.append(z)
	return data, labels

def train():

	t0 = time.time()

	prediction = model(data_placeholder)
	cost = tf.reduce_mean(tf.square(label_placeholder - prediction))
	# cost = tf.reduce_mean(tf.abs(label_placeholder - prediction))
	optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(cost)

	data, labels = prepare_data(100000)
	val_data, val_labels = prepare_data(500)
	batch_size = 2000

	with tf.Session() as sess:

		init = tf.global_variables_initializer()
		init.run()

		saver = tf.train.Saver()

		i = 0
		num = len(data)
		epoch = 0
		num_epoch = 15

		var = [v for v in tf.trainable_variables() if v.name == 'regression_model/output/weights:0']
		print('weights', sess.run(var))

		var = [v for v in tf.trainable_variables() if v.name == 'regression_model/output/biases:0']
		print('biases', sess.run(var))

		while epoch < num_epoch:

			d_batch = []
			l_batch = []
			for _ in range(batch_size):
				d_batch.append(data[i])
				l_batch.append(labels[i])
				i += 1

				if i >= len(data):
					i = 0
					data, labels = shuffle(data, labels)
					epoch += 1
					break

			_, c = sess.run([optimizer, cost], feed_dict={data_placeholder: d_batch, label_placeholder: l_batch})

			if i % 100000 == 0:
				print('cost', c)


		print('TIME TO TRAIN:', time.strftime("%M mins and %S secs", time.gmtime(time.time() - t0)))
		saver.save(sess, 'regression_model/model.ckpt')

		var = [v for v in tf.trainable_variables() if v.name == 'regression_model/output/weights:0']
		print('weights', sess.run(var))

		var = [v for v in tf.trainable_variables() if v.name == 'regression_model/output/biases:0']
		print('biases', sess.run(var))


def inference(a, b):

	prediction = model(data_placeholder)

	with tf.Session() as sess:
		saver = tf.train.Saver()
		saver.restore(sess, 'regression_model/model.ckpt')

		data = [a, b]
		data = np.reshape(data, (1, 2))

		num = sess.run(prediction, feed_dict={data_placeholder: data})
		return num

train()
# print(inference(2, 1))
