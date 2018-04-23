import tensorflow as tf
import numpy as np

from create_sentiment_featureset import create_feature_sets_and_labels

train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')


n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
batch_size = 100

#x is input data
#height x width: no height , and 28*28 width
#height is the number of data we have
#y is label of data
x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float')

#data is input data
def neural_network_model(data):
	#weights are random at first
	#creates a tensor that is 784 width and n_nodes in hl1 as height
	#784 must be width so we can matrix multiply by the x
	#biases don't need to be 2 dimensional, since u just add it at the end
	#and u dont multiply it

	#input data * weights + bias

	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
						'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
						'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
						'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
						'biases':tf.Variable(tf.random_normal([n_classes]))}


	#input data * weighs + bias
	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	#using the relu activation function
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

	return output



#x = input data
def train_neural_network(x):
	prediction = neural_network_model(x)
	#prediction is the one hot array
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
		logits = prediction,
		 labels = y)
	)
	#diff between prediction and the actual label which is y

	# we want to minimize cost, adamoptimizer is basically stochastic gradient descent
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	#default learning rate is 0.001

	#how many epochs, epoch = cycle of feed forward + backwards prop
	hm_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		#nested for loops are just training that network

		for epoch in range(hm_epochs):
			epoch_loss = 0

			#dividing num examples by batch size tells
			#us how many times we must cycle
			
			i = 0

			while i < len(train_x):
				start = i
				end = i + batch_size

				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])

				#c is cost, we are optimizing cost with x's and y's
				#optimizing the cost by modifying the weights "magically"
				_, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
				epoch_loss += c

				i += batch_size

			print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)


		#weights are now optimized

		#argmax returns the index of the maximum value
		#if the prediction equals actual, puts true or false into correct
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

		#cast correct to float
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))


train_neural_network(x)