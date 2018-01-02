from restore_model import predictNum
from addnum import make_prediction
import tensorflow as tf

while True:

	with tf.variable_scope('', reuse = tf.AUTO_REUSE):
		print('Input two file paths with images and returns the sum of the numbers')
		x = input('Enter the image path of the first number\ntest_imgs/')
		x = predictNum('test_imgs/' + str(x))
		y = input('Enter the image path of the second number\ntest_imgs/')
		y = predictNum('test_imgs/' + str(y))
		print('Got the nums')

		z = make_prediction(x, y)
		print('%d + %d = %d' % (x, y, z))