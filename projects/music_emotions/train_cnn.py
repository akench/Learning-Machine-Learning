import tensorflow as tf
import tensorflow.contrib.slim as slim
import pickle
from utils.data_utils import DataUtil
import time
import numpy as np

MODEL_NAME = 'emot_model'
NUM_CLASSES = 6
#128 * 128 = 16384
data_placeholder = tf.placeholder(shape = [None, 16384], dtype = tf.float32)
label_placeholder = tf.placeholder(shape = [None], dtype = tf.int64)
keep_prob_placeholder = tf.placeholder(shape = (), dtype = tf.float32)

logs_path = '/tmp/emot'
#command to use TENSORBOARD
#tensorboard --logdir=run1:/tmp/emot/ --port 6006


def model(data, keep_prob):

    net = tf.reshape(data, (-1, 128, 128, 1))
    tf.summary.image('inputimg', net, 20)

    with tf.variable_scope(MODEL_NAME):
        with slim.arg_scope([slim.conv2d], padding='SAME', stride = 2, activation_fn = tf.nn.relu,
            weights_initializer=tf.contrib.layers.variance_scaling_initializer(uniform=False), 
            weights_regularizer=slim.l2_regularizer(0.05)):

            with slim.arg_scope([slim.fully_connected], 
                weights_initializer=tf.contrib.layers.xavier_initializer(), 
                weights_regularizer=slim.l2_regularizer(0.05)):

                net = slim.conv2d(net, 500, [3,3], scope='conv1')
                net = slim.max_pool2d(net, [2,2], scope='pool1')
                net = slim.conv2d(net, 350, [3,3], scope='conv2')
                net = slim.max_pool2d(net, [2,2], scope='pool2')
                net = slim.conv2d(net, 225, [3,3], scope='conv3')
                net = slim.max_pool2d(net, [2,2], scope='pool3')
                net = slim.conv2d(net, 150, [3,3], scope='conv4')
                net = slim.max_pool2d(net, [2,2], scope='pool4')
                net = slim.conv2d(net, 150, [3,3], scope='conv5')
                net = slim.max_pool2d(net, [2,2], scope='pool5')

                net = slim.flatten(net, scope='flatten7')
                net = slim.fully_connected(net, 500, activation_fn=tf.nn.sigmoid, scope='fc8')
                net = slim.dropout(net, keep_prob=keep_prob, scope='dropout8')

                net = slim.fully_connected(net, NUM_CLASSES, activation_fn=None, scope='fc9')

    return net


def train():

    data_util = DataUtil('processed_data', batch_size = 128, num_epochs = 10)

    prediction = model(data_placeholder, keep_prob_placeholder)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=prediction,
        labels=label_placeholder
    ))

    optimizer = tf.train.AdamOptimizer().minimize(loss)

    num_correct = tf.equal(tf.argmax(prediction, 1), label_placeholder)
    accuracy = tf.reduce_mean(tf.cast(num_correct, 'float'))

    tf.summary.scalar("LOSS", loss)
    tf.summary.scalar("ACCURACY", accuracy)

    summary_op = tf.summary.merge_all()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        t0 = time.time()
        saver = tf.train.Saver()

        writer = tf.summary.FileWriter(logs_path, sess.graph)
        tf.train.write_graph(sess.graph_def, 'out', MODEL_NAME + '.pbtxt', True)

        while data_util.curr_epoch < data_util.num_epochs:


            img_batch, labels_batch, finished_epoch = data_util.get_next_batch()

            if finished_epoch:
                save_path = saver.save(sess, 'out/' + MODEL_NAME + '.chkp')
                print("path saved in", save_path)



            _, summary = sess.run([optimizer, summary_op],
                    feed_dict= {data_placeholder: img_batch,
                                label_placeholder: labels_batch,
                                keep_prob_placeholder: 0.5})


            writer.add_summary(summary, data_util.global_num)


            if data_util.global_num % 100 == 0:

                acc = accuracy.eval({data_placeholder: img_batch,
                                    label_placeholder: labels_batch,
                                    keep_prob_placeholder: 1.0})
                print('curr acc =', acc)


        print('\n\nfinal Accuracy:',accuracy.eval({data_placeholder: data_util.images_val,
                            labels_placeholder: data_util.labels_val,
                            keep_prob_placeholder: 1.0}))

        


    print('TIME TO TRAIN:', time.strftime("%M mins and %S secs", time.gmtime(time.time() - t0)))

    save_path = saver.save(sess, 'out/' + MODEL_NAME + '.chkp')
    print("path saved in", save_path)


    


def main():
    train()



if __name__=='__main__':
    main()
