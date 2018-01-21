import tensorflow as tf
from random import *
from serialize_data import *


def main():

	

	d, l = read_and_decode_single_example('/tmp/add_data')
	d_batch, l_batch = tf.train.shuffle_batch([d, l], batch_size = 128, capacity = 2000, min_after_dequeue=1000)

	d_batch = tf.convert_to_tensor(d_batch, dtype=tf.int32)

	w = tf.get_variable('w1', [2, 10])
	y_pred = tf.matmul(d_batch, w)
	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_pred, l_batch)

	loss_mean = tf.reduce_mean(loss)

	train_op = tf.train.AdamOptimizer().minimize(loss)


	with tf.Session() as sess:
		init = tf.initialize_all_variables()
		sess.run(init)

		tf.train.start_queue_runners(sess=sess)
		d_real, l_real = sess.run([d_batch, l_batch])

		while(True):
			_, loss_val=sess.run([train_op, loss_mean])
			print(loss_val)





def make_data(num):

	all_data=[]
	all_labels=[]

	for _ in range(num):

		a = randint(0,9)
		b = randint(0,9)

		nums = [a,b]
		all_data.append(nums)
		all_labels.append(a+b)

	return all_data, all_labels


data, labels = make_data(10000)
serialize(all_data = data, labels = labels, file_path = '/tmp/add_data', data_purpose=None)
main()