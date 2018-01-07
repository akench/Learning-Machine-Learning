import tensorflow as tf
from tf.add_mnist.train_mnist_cnn import predict_num
from tf.add_mnist.restore_add import predict_sum


while True:

	with tf.variable_scope('', reuse = tf.AUTO_REUSE):
		print('Input two file paths with images and returns the sum of the numbers')
		x = input('Enter the image path of the first number\n../../test_imgs/')
		x = predict_num('../../test_imgs/' + str(x))
		y = input('Enter the image path of the second number\n../../test_imgs/')
		y = predict_num('../../test_imgs/' + str(y))
		print('Got the nums')
		print(x, y)
		z = predict_sum(x, y)
		print('RESULT: %d + %d = %d' % (x, y, z))
