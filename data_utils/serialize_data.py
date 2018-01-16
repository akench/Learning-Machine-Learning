from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from model_utils.model import ModelMonitor
from model_utils import helpers

mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
images = mnist.train.images
labels = mnist.train.labels


def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize(all_data, labels, file_path, data_purpose):
	with tf.python_io.TFRecordWriter(file_path) as w:

		for data, label in zip(all_data, labels):

			feature = {'label': _int64_feature(label),  'data': _bytes_feature(data.tobytes())}

			example = tf.train.Example(features=tf.train.Features(feature=feature))

			w.write(example.SerializeToString())

		print('done serializing')




def deserialize(file_path, data_purpose, batch_size):
	
	with tf.Session() as sess:
	    feature = {data_purpose + '/data' : tf.FixedLenFeature([], tf.string),
	               data_purpose + '/label' : tf.FixedLenFeature([], tf.int64)}

	    filename_queue = tf.train.string_input_producer([file_path], num_epochs=1)

	    reader = tf.TFRecordReader()
	    _, serialized_example = reader.read(filename_queue)

	    features = tf.parse_single_example(serialized_example, features=feature)

	    image = tf.decode_raw(features[data_purpose + '/data'], tf.float32)
	    
	    # Cast label data into int32
	    label = tf.cast(features[data_purpose + '/label'], tf.int32)
	    # Reshape image data into the original shape
	    image = tf.reshape(image, [28, 28])
	    
	    # Any preprocessing here ...
	    
	    # Creates batches by randomly shuffling tensors
	    images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=3*batch_size, 
	    	num_threads=1, min_after_dequeue=batch_size)


	    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	    sess.run(init_op)

	    coord = tf.train.Coordinator()

	    threads = tf.train.start_queue_runners(coord=coord)

	    img, lbl = sess.run([images, labels])

	    coord.request_stop()
	    
	    coord.join(threads)


# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# x = mnist.train.images
# y = mnist.train.labels

# serialize(all_data = x, labels = y, file_path = 'test/test.tfrecords', data_purpose='train')

