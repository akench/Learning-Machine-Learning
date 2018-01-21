import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from random import *
import time
from sklearn.utils import shuffle


data_placeholder = tf.placeholder(shape=[None, 2], dtype = tf.float32)
label_placeholder = tf.placeholder(shape=[None], dtype = tf.float32)

logs_path = '/tmp/add'


def model(data):

	# data = tf.reshape(data, [-1, 2])

	with tf.variable_scope('add_model'):

		with tf.name_scope('weights'):
			weights = tf.Variable([2, 1])
		with tf.name_scope('biases'):
			biases = tf.Variable([1])


		output = tf.add(tf.matmul(data, weights), biases)



		model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'add_model')
		print(model_vars)

		with tf.name_scope('output_node'):
			tf.summary.histogram('weights', model_vars[0])
			tf.summary.histogram('bias', model_vars[1])
			tf.summary.histogram('output', output)



	return output


def prepare_data(num):

	data = []
	labels = []
	for _ in range(num):
		x = randint(0, 9)
		y = randint(0, 9)
		z = x + y
		temp = [x, y]
		data.append(temp)
		labels.append(z)
	return data, labels





def train():

	t0 = time.time()

	prediction = model(data_placeholder)

	with tf.name_scope('cost'):
		cost = tf.reduce_mean(tf.abs(label_placeholder - prediction))


	optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.000001).minimize(cost)

	data, labels = prepare_data(100000)

	print(data[100], labels[100])
	print(data[74], labels[74])

	val_data, val_labels = prepare_data(500)
	batch_size = 1000


	tf.summary.scalar('Cost', cost)
	summary_op = tf.summary.merge_all()

	with tf.Session() as sess:

		init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		sess.run(init)
		saver = tf.train.Saver()

		writer = tf.summary.FileWriter(logs_path, sess.graph)


		i = 0
		num = len(data)
		epoch = 0
		num_epoch = 30
		global_num = 0


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

			with tf.name_scope('current_cost'):
				_, summary = sess.run([optimizer, summary_op], feed_dict={data_placeholder: d_batch, 
														label_placeholder: l_batch})

			writer.add_summary(summary, global_num)


			global_num += 1


		print('TIME TO TRAIN:', time.strftime("%M mins and %S secs", time.gmtime(time.time() - t0)))
		saver.save(sess, 'regression_model/model.ckpt')



def inference(a, b):

	prediction = model(data_placeholder)

	with tf.Session() as sess:
		saver = tf.train.Saver()
		saver.restore(sess, 'regression_model/model.ckpt')

		data = np.reshape([a, b], (1, 2))

		num = sess.run(prediction, feed_dict={data_placeholder: data})
		return num


def main():

	import os
	import glob
	files = glob.glob('/tmp/add/*')
	for f in files:
		os.remove(f)

	train()


# main()
# print(inference(2,10))

# x, y = prepare_data(100)
# print(x[90], y[90])