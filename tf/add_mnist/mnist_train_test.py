import tensorflow.contrib.slim as slim
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from random import *
import time
from cv_stuff.parse_img import resize

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


data_placeholder = tf.placeholder(shape=[None, 784],dtype=tf.float32, name = 'data_placeholder')
label_placeholder = tf.placeholder(shape=[None],dtype=tf.int64)


def CNN_model(net):

	#None doesnt work for some reason, use -1
	net = tf.reshape(net,[-1, 28, 28, 1])

	#defines a scope for each set of weights and biases, so they can be accessed later
	with tf.variable_scope('mnist_model'):
		with slim.arg_scope([slim.conv2d], padding='SAME', weights_initializer=tf.contrib.layers.variance_scaling_initializer(uniform = False), weights_regularizer=slim.l2_regularizer(0.05)):
			with slim.arg_scope([slim.fully_connected], weights_initializer=tf.contrib.layers.variance_scaling_initializer(uniform = False), weights_regularizer=slim.l2_regularizer(0.05)):
				net = slim.conv2d(net, 20, [5,5], scope='conv1')
				net = slim.max_pool2d(net, [2,2], scope='pool1')
				net = slim.conv2d(net, 50, [5,5], scope='conv2')
				net = slim.max_pool2d(net, [2,2], scope='pool2')
				net = slim.flatten(net, scope='flatten3')
				net = slim.fully_connected(net, 500, scope='fc4')
				# net = slim.dropout(net, 0.5, scope='dropout4')
				net = slim.fully_connected(net, 10, activation_fn=None, scope='fc5')
	return net



def remove5to9(x, y):
	new_x = []
	new_y = []

	for rx, ry in zip(x, y):
		if(0 <= np.argmax(ry) <= 4):
			new_x.append(rx)
			new_y.append(ry)
			# new_y.append(np.argmax(ry))
	return new_x, new_y

def remove0to4(x, y):
	new_x = []
	new_y = []

	for rx, ry in zip(x, y):
		if(5 <= np.argmax(ry) <= 9):
			new_x.append(rx)
			new_y.append(ry)
	return new_x, new_y

def prepareFullData(x,y):
	new_x = []
	new_y = []
	for rx, ry in zip(x, y):
		new_x.append(rx)
		new_y.append(ry)
	return new_x, new_y

def shuffle_img_and_labels(x, y):

	for _ in range(10000):
		i = randint(0, len(x)-1)
		j = randint(0, len(x)-1)
		x[i], x[j] = x[j], x[i]
		y[i], y[j] = y[j], y[i]

	return x, y

def scale_up_mnist(images):

	scaled_images = []
	for image in images:
		scaled_image = []
		for pixel in image:
			scaled_image.append(pixel * 255)
		scaled_images.append(scaled_image)

	return scaled_images


def train():

	images = mnist.train.images
	labels = mnist.train.labels



	images_test = mnist.test.images
	labels_test = [np.argmax(l) for l in mnist.test.labels]



	start_time = time.time()



	prediction = CNN_model(data_placeholder)
	total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction,labels=label_placeholder))
	optimizer = tf.train.AdamOptimizer(learning_rate=.001)

	correct = tf.equal(tf.argmax(prediction, 1), label_placeholder)
	accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
	train_step = optimizer.minimize(total_loss)



	with tf.Session() as sess:

		saver = tf.train.Saver()
		init = tf.initialize_all_variables()
		init.run()

		i = 0
		num_epochs = 5
		curr_epoch = 1

		try:
			while curr_epoch <= num_epochs:


				if i >= len(images) or i >= len(labels):
					print('=============================' + str(curr_epoch).upper() + '=============================')
					i = 0
					curr_epoch += 1
					images, labels = shuffle_img_and_labels(images, labels)
					save_path = saver.save(sess, "mnist_model/model.ckpt")
					print('path saved in', save_path)


				batch_xs = []
				batch_ys = []

				for _ in range(500):

					if(i >= len(images) or i >= len(labels)):
						break

					batch_xs.append(images[i])
					batch_ys.append(labels[i])
					i+= 1

				batch_ys = [np.argmax(v) for v in batch_ys]

				sess.run(train_step, feed_dict={data_placeholder: batch_xs, label_placeholder: batch_ys})

				print('current Accuracy:',accuracy.eval({data_placeholder:batch_xs, label_placeholder:batch_ys}))


			print('final Accuracy:',accuracy.eval({data_placeholder:images_test, label_placeholder:labels_test}))
			print('TIME TO TRAIN:', time.strftime("%M mins and %S secs", time.gmtime(time.time() - start_time)))

			save_path = saver.save(sess, "mnist_model/model.ckpt")
			print("path saved in", save_path)


		except KeyboardInterrupt:
			print('TRAINING STOPPED')
			# save_path = saver.save(sess, "mnist_model/model.ckpt")
			# print("path saved in", save_path)
			# print('final Accuracy:',accuracy.eval({data_placeholder:images_test, label_placeholder:labels_test}))



train()