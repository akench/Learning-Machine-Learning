import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from PIL import Image
# from parseImg import resize
from cv_stuff.parseImg import resize

tf.reset_default_graph()
data_placeholder = tf.placeholder(shape=[None, 784], dtype=tf.float32, name = 'data_placeholder')


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
				net = slim.fully_connected(net, 10, activation_fn=None, scope='fc5')
	return net



def predictNum(file_path):

	prediction = CNN_model(data_placeholder)
	print('INSIDE METHOD')
	#the with keyword will automagically close the session after it exits the block
	with tf.Session() as sess:
		# saver = tf.train.import_meta_graph('model.ckpt.meta')
		var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='mnist_model')

		saver = tf.train.Saver(var_list)
		saver.restore(sess, 'mnist_model/mnistmodel.ckpt')
		# print('RESTORED MODEL')
		img = resize(file_path)
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


# print(predictNum('test_imgs/0_test_resized.jpg'))