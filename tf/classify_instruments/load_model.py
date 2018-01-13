import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from cv_stuff.parse_img import resize_crop, images_to_arrays, normalize_data
import pickle
import PIL.ImageOps
from PIL import Image
import glob
from matplotlib import pyplot as plt
from random import *

data_placeholder = tf.placeholder(shape = [None, 784], dtype = tf.float32)
label_placeholder = tf.placeholder(shape=[None], dtype = tf.int64)
keep_prob_placeholder = tf.placeholder(shape = (), dtype = tf.float32, name='keep_prob')

def model(net, keep_prob):
	net = tf.reshape(net, [-1, 28, 28, 1])

	with tf.variable_scope('pvg_model'):
		with slim.arg_scope([slim.conv2d], padding='SAME', weights_initializer=tf.contrib.layers.variance_scaling_initializer(uniform = False), weights_regularizer=slim.l2_regularizer(0.05)):
			with slim.arg_scope([slim.fully_connected], weights_initializer=tf.contrib.layers.variance_scaling_initializer(uniform = False), weights_regularizer=slim.l2_regularizer(0.05)):
				net = slim.conv2d(net, 20, [5,5], scope='conv1')
				net = slim.max_pool2d(net, [2,2], scope='pool1')
				net = slim.conv2d(net, 50, [5,5], scope='conv2')
				net = slim.max_pool2d(net, [2,2], scope='pool2')
				net = slim.conv2d(net, 50, [5,5], scope='conv3')
				net = slim.max_pool2d(net, [2,2], scope='pool3')
				net = slim.flatten(net, scope='flatten4')
				net = slim.fully_connected(net, 500, activation_fn = tf.nn.relu, scope='fc5')
				net = slim.dropout(net, keep_prob = keep_prob, scope='dropout5')
				net = slim.fully_connected(net, 2, activation_fn=None, scope='fc6')
	return net


def make_prediction(data, is_file_path):

	if is_file_path:
		# data = Image.open(data)
		# data = data.resize((28, 28))
		# data = data.convert('L')

		data = resize_crop(img = data, crop_type = 'center', size = 28)
		data.save('org_data/test.jpg')
		data = images_to_arrays([data])

		d = pickle.load(open('processed_data/train_data.p', 'rb'))
		_, m, std = normalize_data(d)

		data -= m
		data /= std

	prediction = model(data_placeholder, keep_prob_placeholder)

	with tf.Session() as sess:

		saver = tf.train.Saver()
		saver.restore(sess, 'out/pvg_model.chkp')

		# var = [v for v in tf.trainable_variables() if v.name == "pvg_model/conv1/weights:0"]
		# print(sess.run(var)[0][0][0][0][0])


		logits = sess.run(prediction, feed_dict={data_placeholder: data, keep_prob_placeholder: 1.0})
		logits = tf.squeeze(logits)
		logits_arr = sess.run(logits)

		print('LOGITS =', logits_arr)
		# if abs(logits_arr[0] - logits_arr[1]) < 100:
		# 	return 'neither'

		softmax_output = tf.nn.softmax(logits = logits_arr)

		n = sess.run(tf.argmax(softmax_output))

		#
		if n == 0:
			return 'piano keyboard'
		else:
			return 'acoustic guitar'
		return 'hi'



def accuracy_on_test_data(test_data, test_labels):

	prediction = model(data_placeholder, keep_prob_placeholder)

	correct = tf.equal(tf.argmax(prediction, 1), label_placeholder)
	accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

	with tf.Session() as sess:

		saver = tf.train.Saver()
		saver.restore(sess, 'out/pvg_model.chkp')

		acc = accuracy.eval({data_placeholder: test_data, label_placeholder: test_labels,
							keep_prob_placeholder: 1.0})

		return acc

def create_test_data_and_labels(folder_name, label):
	paths = glob.glob('org_data/' + folder_name + '/*')
	imgs = []
	for path in paths:
		imgs.append(resize_crop(path))

	data = images_to_arrays(imgs)

	pickle.dump(data, open('processed_data/' + folder_name + '_data.p', 'wb'))

	labels = np.full(len(data), label)
	pickle.dump(labels, open('processed_data/' + folder_name + '_labels.p', 'wb'))

create_test_data_and_labels('my_test', 0)
i = pickle.load(open('processed_data/my_test_data.p', 'rb'))
l = pickle.load(open('processed_data/my_test_labels.p', 'rb'))
i, _, _ = normalize_data(i)
print(accuracy_on_test_data(i, l))
