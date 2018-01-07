from tf.add_mnist.restore_mnist import predictNum
from tf.add_mnist.restore_add import make_prediction
import tensorflow as tf

while True:

	with tf.variable_scope('', reuse = tf.AUTO_REUSE):
		print('Input two file paths with images and returns the sum of the numbers')
		x = input('Enter the image path of the first number\n../../test_imgs/')
		x = predictNum('../../test_imgs/' + str(x))
		y = input('Enter the image path of the second number\n../../test_imgs/')
		y = predictNum('../../test_imgs/' + str(y))
		print('Got the nums')

		z = make_prediction(x, y)
		print('RESULT: %d + %d = %d' % (x, y, z))