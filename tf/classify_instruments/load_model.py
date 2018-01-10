import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from cv_stuff.parse_img import resize_crop, images_to_arrays
import pickle
import PIL.ImageOps
from PIL import Image

data_placeholder = tf.placeholder(shape = [None, 784], dtype = tf.float32)

def model(net):
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
				net = slim.dropout(net, 0.5, scope='dropout5')
				net = slim.fully_connected(net, 2, activation_fn=None, scope='fc6')
	return net


def make_prediction(data, is_file_path):

	if is_file_path:
		data = Image.open(data)
		data = data.resize((28, 28))
		data = data.convert('L')

		# data = resize_crop(img = data, crop_type = 'center', size = 28)
		data.save('org_data/test.jpg')
		data = images_to_arrays([data])

	prediction = model(data_placeholder)

	with tf.Session() as sess:

		# init = tf.global_variables_initializer()
		# init.run()
		# var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'pvg_model')
		# # print(var_list)
		# saver = tf.train.Saver(var_list)

		saver = tf.train.Saver()
		saver.restore(sess, 'model/model.ckpt')

		var = [v for v in tf.trainable_variables() if v.name == "pvg_model/conv1/weights:0"]
		print(sess.run(var)[0][0][0][0][0])


		logits = sess.run(prediction, feed_dict={data_placeholder: data})
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



def accuracy_on_test_data():
	test_data = pickle.load(open('processed_data/test_data.p', 'rb'))



# print(make_prediction('org_data/g1.jpg', is_file_path = True))
# quit()


print(make_prediction('org_data/guitar/fullguitar.jpg', is_file_path = True))
