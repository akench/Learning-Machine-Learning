import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from random import *
import time


data_placeholder = tf.placeholder(shape=[None, 2], dtype = tf.float32)
label_placeholder = tf.placeholder(shape=[None], dtype = tf.int64)
batch_size = 10000

def Model(data):

	data = tf.reshape(data, [-1, 2])
	with tf.variable_scope('add_nums_model'):
		with slim.arg_scope([slim.fully_connected], weights_initializer=tf.contrib.layers.variance_scaling_initializer(uniform = False), weights_regularizer=slim.l2_regularizer(0.05)):
			data = slim.fully_connected(data, 5, activation_fn=tf.nn.sigmoid, scope='firsthidden')
			data = slim.fully_connected(data, 19, activation_fn=None, scope='output')
			# data = slim.fully_connected(data, 1, activation_fn = None, scope = 'output')
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


def train_NN():

	start_time = time.time()
	time_of_last_save = time.time()

	prediction = Model(data_placeholder)
	cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits=prediction,
		labels=label_placeholder)
	)

	optimizer = tf.train.AdamOptimizer().minimize(cost)
	correct = tf.equal(tf.argmax(prediction, 1), label_placeholder)
	accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
	# train_step = optimizer.minimize(cost)

	saver = tf.train.Saver()

	with tf.Session() as sess:

		init = tf.global_variables_initializer()
		init.run()
		init = tf.local_variables_initializer()
		init.run()


		data, labels = prepare_data(10000000)
		acc = 0.0
		i = 0

		while acc < 0.99:

			batch_d = []
			batch_l = []

			for _ in range(batch_size):
				i = randint(0, len(data) - 1)
				batch_d.append(data[i])
				batch_l.append(labels[i])

			_, c = sess.run([optimizer, cost], feed_dict = {data_placeholder: batch_d, label_placeholder: batch_l})

			acc = accuracy.eval({data_placeholder: batch_d, label_placeholder: batch_l})

			if i % 100 == 0:
				print('curr acc=', acc)
				print('curr loss=', c)

			i += 1

			if time.time() - time_of_last_save >= 300:
				save_path = saver.save(sess, "add_nums_model/model.ckpt")
				print("path saved")
				time_of_last_save = time.time()



		test_data, test_labels = prepare_data(100)
		print('final accuracy = ', accuracy.eval({data_placeholder: test_data, label_placeholder: test_labels}))
		save_path = saver.save(sess, "add_nums_model/model.ckpt")
		print("path saved in '/add_nums_model/model.ckpt'")
		print('Time to train: ', time.time() - start_time)

def predict_sum(a, b):
	# print('INSIDE MAKE PREDICTION')
	prediction = Model(data_placeholder)

	with tf.Session() as sess:

		var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'add_nums_model')
		saver = tf.train.Saver(var_list)
		saver.restore(sess, 'add_nums_model/model.ckpt')

		data = [a, b]
		data = np.reshape(data, (1, 2))
		logits = sess.run(prediction, feed_dict = {data_placeholder: data})
		# print('logits ', logits)
		sftmx = tf.nn.softmax(logits = tf.squeeze(logits))
		# print(sess.run(sftmx))


		var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
		var = [v for v in tf.trainable_variables() if v.name == "add_nums_model/firsthidden/weights:0"][0]
		# print(sess.run(var))

		return sess.run(tf.argmax(sftmx))

# with tf.variable_scope('') as scope:
# 	while True:
# 		choice = input('Enter "t" to train network, or enter "a" to add numbers, or "q" to quit \n')
# 		if choice == 't':
# 			train_NN()
#
# 		elif choice == 'a':
# 			a = input('Enter first num: ')
# 			b = input('Enter second num: ')
# 			print('sum = ', make_prediction(a,b))
# 			scope.reuse_variables()
#
# 		elif choice == 'q':
# 			quit()
# 		else:
# 			print('Please enter a valid input')
