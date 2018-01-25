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
from utils.parse_img import normalize_data


MODEL_NAME = 'pvg_model'
data_placeholder = tf.placeholder(shape = [None, 784], dtype = tf.float32, name='input')
labels_placeholder = tf.placeholder(shape = [None], dtype = tf.int64)
keep_prob_placeholder = tf.placeholder(shape = (), dtype = tf.float32, name='keep_prob')

logs_path = "/tmp/instr"
#command to use TENSORBOARD
#tensorboard --logdir=run1:/tmp/instr/ --port 6006

import os
import glob

files = glob.glob('/tmp/instr/test/*')
files += glob.glob('/tmp/instr/train/*')

for f in files:
	os.remove(f)



class DataUtil:

	def __init__(self, batch_size, num_epochs):
		self.images_train = list(pickle.load(open('processed_data/train_data.p', 'rb')))
		self.labels_train = list(pickle.load(open('processed_data/train_labels.p', 'rb')))

		images_val = list(pickle.load(open('processed_data/val_data.p', 'rb')))
		images_val_norm, _, _ = normalize_data(images_val)
		self.images_val_norm = images_val_norm
		self.labels_val = list(pickle.load(open('processed_data/val_labels.p', 'rb')))

		self.batch_size = batch_size
		self.curr_data_num = 0
		self.global_num = 0

		self.curr_epoch = 0
		self.num_epochs = num_epochs


	def get_next_batch(self):
		'''
		Gets the next batch in training data.
		@param None
		@return The next normalized training batch
		'''

		img_batch = []
		labels_batch = []

		for _ in range(self.batch_size):

			img_batch.append(self.images_train[self.curr_data_num])
			labels_batch.append(self.labels_train[self.curr_data_num])

			self.curr_data_num += 1
			self.global_num += 1

			if self.curr_data_num > len(self.images_train) - 1:

				print('FINISHED EPOCH', self.curr_epoch)
				self.curr_epoch += 1
				self.curr_data_num = 0
				self.images_train, self.labels_train = shuffle(self.images_train, self.labels_train)


		img_batch, _, _ = normalize_data(img_batch)

		return img_batch, labels_batch





def model(net, keep_prob):
	net = tf.reshape(net, [-1, 28, 28, 1])

	tf.summary.image('input', net, 10)

	with tf.variable_scope(MODEL_NAME):
		with slim.arg_scope([slim.conv2d], padding='SAME', weights_initializer=tf.contrib.layers.variance_scaling_initializer(uniform = False), weights_regularizer=slim.l2_regularizer(0.05)):
			with slim.arg_scope([slim.fully_connected], weights_initializer=tf.contrib.layers.variance_scaling_initializer(uniform = False), weights_regularizer=slim.l2_regularizer(0.05)):
				net = slim.conv2d(net, 20, [5,5], scope='conv1')
				net = slim.max_pool2d(net, [2,2], scope='pool1')
				net = slim.conv2d(net, 50, [5,5], scope='conv2')
				net = slim.max_pool2d(net, [2,2], scope='pool2')
				net = slim.conv2d(net, 100, [5,5], scope='conv3')
				net = slim.max_pool2d(net, [2,2], scope='pool3')
				net = slim.flatten(net, scope='flatten4')
				net = slim.fully_connected(net, 500, activation_fn = tf.nn.sigmoid, scope='fc5')
				net = slim.dropout(net, keep_prob = keep_prob, scope='dropout6')
				net = slim.fully_connected(net, 4, activation_fn=None, scope='fc6')

				fc6_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'pvg_model/fc6')

				with tf.name_scope('fc6'):
					tf.summary.histogram('weights', fc6_vars[0])
					tf.summary.histogram('bias', fc6_vars[1])
					tf.summary.histogram('output', net)

	output = tf.identity(net, name='output')
	# outputs = tf.nn.softmax(net, name='output')
	return net




def train():

	data_util = DataUtil(batch_size = 128, num_epochs = 7)



	prediction = model(data_placeholder, keep_prob_placeholder)

	with tf.name_scope('total_loss'):
		total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
				logits = prediction, labels = labels_placeholder))

	with tf.name_scope('train_step'):
		optimizer = tf.train.AdamOptimizer()
		train_step = optimizer.minimize(total_loss)

	with tf.name_scope('accuracy'):
		correct = tf.equal(tf.argmax(prediction, 1), labels_placeholder)
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

	tf.summary.scalar("total_loss", total_loss)
	tf.summary.scalar("accuracy", accuracy)
	# tf.summary.scalar("train_step", optimizer)




	#MERGES ALL SUMMARIES INTO ONE OPERATION
	#THIS CAN BE EXECUTED IN A SESSION
	summary_op = tf.summary.merge_all()

	with tf.Session() as sess:
		start_time = time.time()
		saver = tf.train.Saver()
		init = tf.global_variables_initializer()
		init.run()

		#FOR TENSORBOARD, CREATES A LOG WRITER OBJECT
		# writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
		train_writer = tf.summary.FileWriter(logs_path + '/train', sess.graph)
		test_writer = tf.summary.FileWriter(logs_path + '/test', sess.graph)

		tf.train.write_graph(sess.graph_def, 'out', MODEL_NAME + '.pbtxt', True)

		while data_util.curr_epoch <= data_util.num_epochs:

			'''VALIDATION ACCURACY'''
			if data_util.global_num % 30 == 0:
				# print('inside val')
				with tf.name_scope('val_acc'):
					_, summary_val = sess.run([accuracy, summary_op],
									feed_dict = {data_placeholder: data_util.images_val_norm,
									labels_placeholder: data_util.labels_val,
									keep_prob_placeholder: 1.0})
				test_writer.add_summary(summary_val, data_util.global_num)



			'''ACTUAL TRAINING'''
			img_batch, labels_batch = data_util.get_next_batch()

			_, summary = sess.run([train_step, summary_op],
												feed_dict = {data_placeholder: img_batch,
												labels_placeholder: labels_batch,
												keep_prob_placeholder: 0.5})


			train_writer.add_summary(summary, data_util.global_num)








		with tf.name_scope('val_acc'):
			_, summary_val = sess.run([accuracy, summary_op],
							feed_dict = {data_placeholder: data_util.images_val_norm,
							labels_placeholder: data_util.labels_val,
							keep_prob_placeholder: 1.0})
		test_writer.add_summary(summary_val, data_util.global_num)

		#TRAINING DONE!!!!!!!!!!!!!!
		#VAL IMAGES ALREADY NORMALIZED
		print('\n\nfinal Accuracy:',accuracy.eval({data_placeholder: data_util.images_val_norm,
											labels_placeholder: data_util.labels_val,
											keep_prob_placeholder: 1.0}))



		print('TIME TO TRAIN:', time.strftime("%M mins and %S secs", time.gmtime(time.time() - start_time)))

		save_path = saver.save(sess, 'out/' + MODEL_NAME + '.chkp')
		print("path saved in", save_path)



def export_model(input_node_names, output_node_name):
	freeze_graph.freeze_graph('out/' + MODEL_NAME + '.pbtxt', None, False,
		'out/' + MODEL_NAME + '.chkp', output_node_name, "save/restore_all",
		"save/Const:0", 'out/frozen_' + MODEL_NAME + '.pb', True, "")

	input_graph_def = tf.GraphDef()
	with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
		input_graph_def.ParseFromString(f.read())

	output_graph_def = optimize_for_inference_lib.optimize_for_inference(
			input_graph_def, input_node_names, [output_node_name],
			tf.float32.as_datatype_enum)

	with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
		f.write(output_graph_def.SerializeToString())


	images_train = list(pickle.load(open('processed_data/train_data.p', 'rb')))
	_, mean, std = normalize_data(images_train)
	mean = mean.flatten()
	std = std.flatten()
	with open('popMean.txt', 'w') as f:
		for m in mean:
			f.write(str(m) + "\n")

	with open('popSTD.txt', 'w') as f:
		for s in std:
			f.write(str(s) + "\n")

	print("graph saved!")



def main():
	if not path.exists('out'):
		os.mkdir('out')

	input_node_name = 'input'
	output_node_name = 'output'
	keep_prob_name = 'keep_prob'

	train()

	# export_model([input_node_name, keep_prob_name], output_node_name)

if __name__ == '__main__':
	main()
