import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from PIL import Image
from cv_stuff.parse_img import resize

# tf.reset_default_graph()
data_placeholder = tf.placeholder(shape=[1, 784], dtype=tf.float32, name = 'data_placeholder')


def CNN_model(data):

	#None doesnt work for some reason, use -1
	data = tf.reshape(data,[-1, 28, 28, 1])

	#defines a scope for each set of weights and biases, so they can be accessed later
	with tf.variable_scope('mnist_model'):
		with slim.arg_scope([slim.conv2d], padding='SAME', weights_initializer=tf.contrib.layers.variance_scaling_initializer(uniform = False), weights_regularizer=slim.l2_regularizer(0.05)):
			with slim.arg_scope([slim.fully_connected], weights_initializer=tf.contrib.layers.variance_scaling_initializer(uniform = False), weights_regularizer=slim.l2_regularizer(0.05)):
				net = slim.conv2d(data, 20, [5,5], scope='conv1')
				net = slim.max_pool2d(net, [2,2], scope='pool1')
				net = slim.conv2d(net, 50, [5,5], scope='conv2')
				net = slim.max_pool2d(net, [2,2], scope='pool2')
				net = slim.flatten(net, scope='flatten3')
				net = slim.fully_connected(net, 500, scope='fc4')
				# net = slim.dropout(net, 0.5, scope='dropout4')
				net = slim.fully_connected(net, 10, activation_fn=None, scope='fc5')
	return net



def predict_num(file_path):

	prediction = CNN_model(data_placeholder)
	print('INSIDE METHOD')
	#the with keyword will automagically close the session after it exits the block
	with tf.Session() as sess:
		# saver = tf.train.import_meta_graph('model.ckpt.meta')
		var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='mnist_model')
		print('var_list', var_list)

		saver = tf.train.Saver(var_list)
		saver.restore(sess, 'mnist_model/model.ckpt')
		# print('RESTORED MODEL')

		try:
			img = Image.open(file_path).convert('L')
		except IOError:
			print(file_path)
			print('File not found')
			quit()

		img = img.resize((28, 28), PIL.Image.ANTIALIAS)
		img.save(file_path)

		arr = np.array(img).reshape(1, 784)

		# inverts image
		for i in range(len(arr)):
			arr[i] = 255 - arr[i]

		logits = sess.run(prediction, feed_dict = {data_placeholder : arr})

		#squeeze it to get rid of useless dimensions (ranks)
		sftmx = tf.nn.softmax(logits=tf.squeeze(logits))
		#turns tensor into numpy array
		# sftmx = sftmx.eval()

		num = sess.run(tf.argmax(sftmx))

		# with tf.variable_scope('fc4', reuse=True):
  # 			print(sess.run(tf.get_variable('weights')))

		return num



def predict_num_test(file_path):

	prediction = CNN_model(data_placeholder)
	# print('INSIDE METHOD')
	#the with keyword will automagically close the session after it exits the block
	with tf.Session() as sess:

		saver = tf.train.import_meta_graph('mnist_model/model.ckpt.meta', clear_devices = True)
		tf.global_variables_initializer().run()
		saver.restore(sess, 'mnist_model/model.ckpt')

		# var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='mnist_model')
		# saver = tf.train.Saver(var_list)
		# saver.restore(sess, 'mnist_model/model.ckpt')
		# print('RESTORED MODEL')
		img = resize(file_path)
		org = np.array(img).reshape(1, 784)
		# arr = arr[0]

		arr = org - 255
		# print(arr)

		scaled = []
		for inner in arr:
			for num in inner:
				scaled.append(num / float(255))
		scaled = np.reshape(scaled, (1,784))


		# print(scaled)
		logits = sess.run(prediction, feed_dict = {data_placeholder : scaled})

		#squeeze it to get rid of useless dimensions (ranks)
		sftmx = tf.nn.softmax(logits=tf.squeeze(logits))
		#turns tensor into numpy array
		# sftmx = sftmx.eval()
		num = sess.run(tf.argmax(sftmx))
		# print(sess.run(sftmx))

		# with tf.variable_scope('fc4', reuse=True):
  # 			print(sess.run(tf.get_variable('weights')))

		return num




with tf.variable_scope('', reuse=tf.AUTO_REUSE):
	print(predict_num('../../test_imgs/1_test.jpg'))
	print(predict_num_test('../../test_imgs/2_test.jpg'))
