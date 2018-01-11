import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
import numpy as np
import time
import pickle
from sklearn.utils import shuffle
import os
import os.path as path

data_placeholder = tf.placeholder(shape = [None, 784], dtype = tf.float32, name='input')
labels_placeholder = tf.placeholder(shape = [None], dtype = tf.int64)
keep_prob_placeholder = tf.placeholder(dtype = tf.float32, name='keep_prob')

def model(net, keep_prob):
	net = tf.reshape(net, [-1, 28, 28, 1])

	if keep_prob == 1.0:
		is_training = False
	else:
		is_training = True


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
				net = slim.dropout(net, keep_prob = keep_prob, scope='dropout5', is_training = is_training)
				net = slim.fully_connected(net, 2, activation_fn=None, scope='fc6')
	outputs = tf.nn.softmax(net, name='output')
	return net

def train():
	images_train = list(pickle.load(open('processed_data/train_data.p', 'rb')))
	labels_train = list(pickle.load(open('processed_data/train_labels.p', 'rb')))

	images_val = list(pickle.load(open('processed_data/val_data.p', 'rb')))
	labels_val = list(pickle.load(open('processed_data/val_labels.p', 'rb')))


	prediction = model(data_placeholder, keep_prob_placeholder)

	total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits = prediction, labels = labels_placeholder))

	optimizer = tf.train.AdamOptimizer()
	train_step = optimizer.minimize(total_loss)

	correct = tf.equal(tf.argmax(prediction, 1), labels_placeholder)
	accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

	with tf.Session() as sess:
		start_time = time.time()
		saver = tf.train.Saver()
		init = tf.global_variables_initializer()
		init.run()

		tf.train.write_graph(sess.graph_def, 'out', 'pvg_model.pbtxt', True)

		curr_epoch = 1
		num_epochs = 5
		batch_size = 128
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
					images_train, labels_train = shuffle(images_train, labels_train)

			sess.run(train_step, feed_dict = {data_placeholder: img_batch, 
												labels_placeholder: labels_batch,
												keep_prob_placeholder: 0.5})
			
			print('Current Accuracy:', accuracy.eval({data_placeholder: img_batch, 
										labels_placeholder: labels_batch,
										keep_prob_placeholder: 1.0}))




		print('\n\nfinal Accuracy:',accuracy.eval({data_placeholder: images_val, 
													labels_placeholder: labels_val,
													keep_prob_placeholder: 1.0}))
		print('TIME TO TRAIN:', time.strftime("%M mins and %S secs", time.gmtime(time.time() - start_time)))

		save_path = saver.save(sess, "out/pvg_model.chkp")
		print("path saved in", save_path)



def export_model(input_node_names, output_node_name):
	freeze_graph.freeze_graph('out/pvg_model.pbtxt', None, False,
		'out/pvg_model.chkp', output_node_name, "save/restore_all",
		"save/Const:0", 'out/frozen_pvg_model.pb', True, "")

	input_graph_def = tf.GraphDef()
	with tf.gfile.Open('out/frozen_pvg_model.pb', "rb") as f:
		input_graph_def.ParseFromString(f.read())

	output_graph_def = optimize_for_inference_lib.optimize_for_inference(
			input_graph_def, input_node_names, [output_node_name],
			tf.float32.as_datatype_enum)

	with tf.gfile.FastGFile('out/opt_pvg_model.pb', "wb") as f:
		f.write(output_graph_def.SerializeToString())



def main():
	if not path.exists('out'):
		os.mkdir('out')

	input_node_name = 'input'
	output_node_name = 'output'
	keep_prob_name = 'keep_prob'

	train()

	export_model([input_node_name, keep_prob_name], output_node_name)

if __name__ == '__main__':
	main()