import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from cv_stuff.parse_img import resize_crop, images_to_arrays

data_placeholder = tf.placeholder(shape = [None, 784], dtype = tf.float32)

def model(net):
	net = tf.reshape(net, [-1, 28, 28, 1])

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
				net = slim.fully_connected(net, 500, scope='fc5')
				net = slim.dropout(net, 0.5, scope='dropout5')
				net = slim.fully_connected(net, 2, activation_fn=None, scope='fc6')
	return net


def make_prediction(file_path):

    img = resize_crop(img = file_path, crop_type = 'center', size = 28)
    data = images_to_arrays([img])

    prediction = model(data_placeholder)

    with tf.Session() as sess:
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'pvg_model')
        saver = tf.train.Saver(var_list)
        saver.restore(sess, 'model/model.ckpt')

        logits = sess.run(prediction, feed_dict={data_placeholder: data})
        logits = tf.squeeze(logits)
        print('logits', sess.run(logits))

        softmax_output = tf.nn.softmax(logits = logits)
        print(sess.run(softmax_output))

        n = sess.run(tf.argmax(softmax_output))

        if n == 0:
            return 'piano'
        else:
            return 'guitar'

print(make_prediction('org_data/guitar/g1.jpg'))
