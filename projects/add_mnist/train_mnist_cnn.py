import tensorflow.contrib.slim as slim
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from random import *
from sklearn.utils import shuffle

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
				net = slim.fully_connected(net, 10, activation_fn=None, scope='fc5')
	return net



def prepareFullData(x,y):
	new_x = []
	new_y = []
	for rx, ry in zip(x, y):
		new_x.append(rx)
		new_y.append(ry)
	return new_x, new_y


images = mnist.train.images
labels = mnist.train.labels
print(len(images))


images_test = mnist.test.images
labels_test = [np.argmax(l) for l in mnist.test.labels]
prediction = CNN_model(data_placeholder)
# import pdb;pdb.set_trace()
#slim.losses.softmax_cross_entropy(labels,prediction)
# import pdb;pdb.set_trace()
total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction,labels=label_placeholder))
optimizer = tf.train.AdamOptimizer(learning_rate=.001)

correct = tf.equal(tf.argmax(prediction, 1), label_placeholder)
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
train_step = optimizer.minimize(total_loss)



with tf.Session() as sess:

	# init = tf.global_variables_initializer()
	# init.run()
	# init = tf.local_variables_initializer()
	# init.run()
	saver = tf.train.Saver()

	init = tf.initialize_all_variables()
	init.run()

	i = 0

	for c in range(100):


		if i >= len(images) or i >= len(labels):
			print('starting new epoch')
			i = 0
			images, labels = shuffle(images, labels)


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

	save_path = saver.save(sess, "mnist_model/mnistmodel.ckpt")
	print("path saved in 'mnist_model/mnistmodel.ckpt'")
