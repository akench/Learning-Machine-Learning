from restore_mnist import predict_num
from train_add import predict_sum
import tensorflow as tf

while True:

	with tf.variable_scope('', reuse = tf.AUTO_REUSE):
		print('Input two file paths with images and returns the sum of the numbers')
		x = input('Enter the image path of the first number\ntest_imgs/')
		x = predict_num('test_imgs/' + str(x))
		y = input('Enter the image path of the second number\ntest_imgs/')
		y = predict_num('test_imgs/' + str(y))
		print('Got the nums')

		z = predict_sum(x, y)
		print('%d + %d = %d' % (x, y, z))
