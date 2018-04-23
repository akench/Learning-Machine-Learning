
import tensorflow as tf
import numpy as np




def _float_feature(value, is_list=False):
    if not is_list:
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value, is_list=False):
    if not is_list:
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value, is_list=False):
    if not is_list:
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def serialize(all_data, labels, file_path, data_purpose):

    with tf.python_io.TFRecordWriter(file_path) as w:


        for data, label in zip(all_data, labels):

            # feature = {'train/label': _int64_feature(label),  'train/data': _bytes_feature(tf.compat.as_bytes(data.tostring()))}
            feature = {'train/label' : _int64_feature(label), 'train/data': _int64_feature(data, True)}
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            w.write(example.SerializeToString())

        print('done serializing')




def deserialize(file_path, batch_size, data_dims):
    
    with tf.Session() as sess:
        feature = {'train/data' : tf.FixedLenFeature([], tf.string),
                   'train/label' : tf.FixedLenFeature([], tf.int64)}

        filename_queue = tf.train.string_input_producer([file_path], num_epochs=None)

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(serialized_example, features=feature)

        image = tf.decode_raw(features['train/data'], tf.float32)


        label = tf.cast(features['train/label'], tf.int32)

        image = tf.reshape(image, [784])

        images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=3*batch_size, 
            num_threads=1, min_after_dequeue=batch_size)


        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(coord=coord)

        img, lbl = sess.run([images, labels])

        coord.request_stop()
        coord.join(threads)


        return img, lbl


def read_and_decode_single_example(file_path):

    filename_queue = tf.train.string_input_producer([file_path], num_epochs = None)
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features = {
        'train/data' : tf.FixedLenFeature([], tf.int64),
        'train/label' : tf.FixedLenFeature([], tf.int64)
        })


    data = features['train/data']
    label = features['train/label']

    return data, label


def get_data_batches(batch_size, file_path):

    data, label = read_and_decode_single_example(file_path)

    data_batch, labels_batch = tf.train.shuffle_batch(
        [data, label], batch_size = batch_size, capacity = 2000,
        min_after_dequeue = 1000)

    with tf.Session() as sess:

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)


        coord = tf.train.Coordinator()

        tf.train.start_queue_runners(sess = sess)
        all_data, all_labels = sess.run([data_batch, labels_batch])

       
        threads = tf.train.start_queue_runners(coord=coord)

        img, lbl = sess.run([all_data, all_labels])

        coord.request_stop()
        coord.join(threads)

        return img, lbl






# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
# x = mnist.train.images
# y = mnist.train.labels

# # serialize(all_data = x, labels = y, file_path = '/tmp/test.tfrecords', data_purpose='train')
# # i, l = deserialize('/tmp/test.tfrecords', 100)
# i, l = get_data_batches(128, '/tmp/test.tfrecords')
# print(l[0])