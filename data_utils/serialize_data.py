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

def serialize():
	with tf.python_io.TFRecordWriter('dataset/mnist.tfrecords') as w:

		i = 0
		for image, label in zip(images, labels):

			if i%10000 == 0:
				print(i)

			feature = {'number': _int64_feature(label),  'input': _bytes_feature(np.array(image).tobytes())}
			# feature = {'number': _int64_feature(label),  'input': _bytes_feature(image.tobytes())}

			example = tf.train.Example(features=tf.train.Features(feature=feature))

			w.write(example.SerializeToString())

			i+=1
		print('done serializing')




def deserialize():
	serialized_example = helpers.get_serialized_examples(['mnist.tfrecords'], 1)

	batch = helpers.generate_batch(serialized_example, batch_size = 100, capacity = 10, shuffle = False)

	feature = {'image': tf.FixedLenFeature([], tf.float32),  'label': tf.FixedLenFeature([], tf.int64)}
	deserialized = helpers.tfrecord_deserializer(batch, annotation_map = feature, classes = ['label'], shape = (1, 784, 1), batch_size = 100)
	
	print(deserialized[0])
	return deserialized

serialize()


