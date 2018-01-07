import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from PIL import Image
# from parseImg import resize
from cv_stuff.parseImg import resize

# tf.reset_default_graph()
data_placeholder = tf.placeholder(shape=[1, 2], dtype=tf.float32, name = 'data_placeholder')


def Model(data):

	data = tf.reshape(data, [-1, 2])
	with tf.variable_scope('add_nums_model'):
		with slim.arg_scope([slim.fully_connected], weights_initializer=tf.contrib.layers.variance_scaling_initializer(uniform = False), weights_regularizer=slim.l2_regularizer(0.05)):
			data = slim.fully_connected(data, 5, activation_fn=tf.nn.sigmoid, scope='firsthidden')
			data = slim.fully_connected(data, 19, activation_fn=None, scope='output')
			# data = slim.fully_connected(data, 1, activation_fn = None, scope = 'output')
	return data



def predict_sum(a, b):
	# print('INSIDE MAKE PREDICTION')
	prediction = Model(data_placeholder)

	with tf.Session() as sess:

		var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'add_nums_model')
		print('varlist', var_list)
		saver = tf.train.Saver(var_list)
		saver.restore(sess, 'add_nums_model/model.ckpt')

		data = [a, b]
		data = np.reshape(data, (1, 2))
		logits = sess.run(prediction, feed_dict = {data_placeholder: data})
		# print('logits ', logits)
		sftmx = tf.nn.softmax(logits = tf.squeeze(logits))
		# print(sess.run(sftmx))


		# var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
		# var = [v for v in tf.trainable_variables() if v.name == "add_nums_model/firsthidden/weights:0"][0]
		# print(sess.run(var))

		return sess.run(tf.argmax(sftmx))


# print(make_prediction(4,5))
