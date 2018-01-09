import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time
import pickle
from sklearn.utils import shuffle

data_placeholder = tf.placeholder(shape = [None, 784], dtype = tf.float32)
labels_placeholder = tf.placeholder(shape = [None], dtype = tf.int64)

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


images_train = list(pickle.load(open('processed_data/train_data.p', 'rb')))
labels_train = list(pickle.load(open('processed_data/train_labels.p', 'rb')))

images_val = list(pickle.load(open('processed_data/val_data.p', 'rb')))
labels_val = list(pickle.load(open('processed_data/val_labels.p', 'rb')))


prediction = model(data_placeholder)

total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits = prediction, labels = labels_placeholder))

optimizer = tf.train.AdamOptimizer()
train_step = optimizer.minimize(total_loss)

correct = tf.equal(tf.argmax(prediction, 1), labels_placeholder)
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

with tf.Session() as sess:
	start_time = time.time()
	saver = tf.train.Saver()
	init = tf.initialize_all_variables()
	init.run()

	curr_epoch = 1
	num_epochs = 5
	batch_size = 64
	num_examples = len(images_train)
	img_num = 0

	while curr_epoch <= num_epochs:

		img_batch = []
		labels_batch = []

		for _ in range(batch_size):

			img_batch.append(images_train[img_num])
			labels_batch.append(labels_train[img_num])

			img_num += 1

			if img_num > num_examples - 1:

				print('FINISHED EPOCH', curr_epoch)
				curr_epoch += 1
				img_num = 0
				saver.save(sess, 'model/model.ckpt')
				print('path saved')

				images_train, labels_train = shuffle(images_train, labels_train)

		sess.run(train_step, feed_dict = {data_placeholder: img_batch, labels_placeholder: labels_batch})
		# print('Current Accuracy:', accuracy.eval({data_placeholder: img_batch, labels_placeholder: labels_batch}))


	print('final Accuracy:',accuracy.eval({data_placeholder:images_val, labels_placeholder:labels_val}))
	print('TIME TO TRAIN:', time.strftime("%M mins and %S secs", time.gmtime(time.time() - start_time)))

	save_path = saver.save(sess, "model/model.ckpt")
	print("path saved in", save_path)
