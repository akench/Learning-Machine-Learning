import tensorflow as tf
import os
from generate_data import dl_audio
from make_spectrogram import graph_spectrogram
from PIL import Image
import numpy as np
import utils.parse_img as parse_img


label_to_emot = {0:'happy', 1:'motivational', 2:'relaxing', 3:'angry', 4:'sad', 5:'tense'}


def load_graph(graph_file_path):

	with tf.gfile.GFile(graph_file_path, 'rb') as f:
		#load the pbfile and get the unserialized graph_Def
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	#import this graph_def into a new graph
	with tf.Graph().as_default() as graph:
		tf.import_graph_def(graph_def, name="inceptionv3")

	return graph


def inference(data):

	graph = load_graph('/home/super/Desktop/output_graph.pb')

	x = graph.get_tensor_by_name('inceptionv3/DecodeJpeg:0')
	y = graph.get_tensor_by_name('inceptionv3/final_result:0')

	with tf.Session(graph = graph) as sess:
		out = sess.run(y, feed_dict={x: data})
		label = sess.run(tf.argmax(tf.squeeze(out)))
		return label


def softmax(x):

	e = np.exp(np.array(x))
	dist = e / np.sum(e)
	print(dist)
	return dist


def predict_song(spec_list):

	label_list = [0, 0, 0, 0, 0, 0]

	for img in spec_list:
		
		img = parse_img.resize_crop(img, size=299, grey = False)

		lbl = inference(img)
		print(lbl)
		label_list[lbl] += 1

	print(label_list)

	return softmax(label_list)


def make_output(softmax_output):

	print('The song is:')

	for i in range(len(softmax_output)):
		print('%f percent %s' % (softmax_output[i] * 100, label_to_emot[i]))






if __name__ == '__main__':


	choice = input('perform inference on one [i], or organize your musical library [o]?\n').strip()

	if choice == 'i':
		audio = input('Enter a youtube link or filepath to an mp3\n')

		if 'https' in audio:
			dl_audio_path = dl_audio(audio, None)
			specs = graph_spectrogram(dl_audio_path, 10, save = False)
			os.remove(dl_audio_path)
		else:
			specs = graph_spectrogram(audio, 10, save = False)


		sftmx = predict_song(specs)
		make_output(sftmx)


		

		


	elif choice == 'o':
		directory = input('Enter filepath to directory of mp3 files')





