import tensorflow.contrib.slim as slim
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np 

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#stores checkpts??
train_log_dir = 'log_dir_test'


data_placeholder = tf.placeholder(shape=[None, 784],dtype=tf.float32)
label_placeholder = tf.placeholder(shape=[None],dtype=tf.int64)



if not tf.gfile.Exists(train_log_dir):
	tf.gfile.MakeDirs(train_log_dir)


def CNN_model(data):	

	#None doesnt work for some reason, use -1
	data = tf.reshape(data,[-1, 28, 28, 1])

	#defines a scope for each set of weights and biases, so they can be accessed later
	with slim.arg_scope([slim.conv2d], padding='SAME', weights_initializer=tf.truncated_normal_initializer(stddev=0.9), weights_regularizer=slim.l2_regularizer(0.05)):
		net = slim.conv2d(data, 20, [5,5], scope='conv1')
		net = slim.max_pool2d(net, [2,2], scope='pool1')
		net = slim.conv2d(net, 50, [5,5], scope='conv2')
		net = slim.max_pool2d(net, [2,2], scope='pool2')
		net = slim.flatten(net, scope='flatten3')
		net = slim.fully_connected(net, 500, scope='fc4')
		net = slim.fully_connected(net, 10, activation_fn=None, scope='fc5')
	return net



images = mnist.train.images
labels = mnist.train.labels
images_test = mnist.test.images
labels_test = [np.argmax(l) for l in mnist.test.labels]
prediction = CNN_model(data_placeholder)
# import pdb;pdb.set_trace()
#slim.losses.softmax_cross_entropy(labels,prediction)
total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.squeeze(prediction),labels=label_placeholder))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)
#train_tensor = slim.learning.create_train_op(total_loss, optimizer)
# slim.learning.train(train_tensor, train_log_dir,feed_dict={data_placeholder:images, label_placeholder:labels})

print(total_loss)


correct = tf.equal(tf.argmax(prediction, 1), label_placeholder)
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
train_step = optimizer.minimize(total_loss)
init = tf.global_variables_initializer()
with tf.Session() as sess:
	init.run()
	for c in range(100):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		batch_ys = [np.argmax(v) for v in batch_ys]
		sess.run(train_step, feed_dict={data_placeholder: batch_xs, label_placeholder: batch_ys})

		# if c % 10 == 0:
		# 	print("test")

		print('Accuracy:',accuracy.eval({data_placeholder:batch_xs, label_placeholder:batch_ys}))
	print('final Accuracy:',accuracy.eval({data_placeholder:images_test, label_placeholder:labels_test}))