import tensorflow as tf
import os
from generate_data import dl_audio
from make_spectrogram import graph_spectrogram
from PIL import Image
import numpy as np
import utils.parse_img as parse_img
import glob
import numpy as np
import time


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


def percent(x):

	perc = (x / np.sum(x))
	print(perc)
	return perc

def predict_song(spec_list):

	label_list = [0, 0, 0, 0, 0, 0]

	for img in spec_list:

		img = parse_img.resize_crop(img, size=299, grey = False)

		lbl = inference(img)
		label_list[lbl] += 1

	return label_list


def print_results(output):

	output = percent(output)

	print('The song is:')

	for i in range(len(output)):
		if output[i] > 0.001:
			print('%f percent %s' % (output[i] * 100, label_to_emot[i]))


def emot_to_new_path(old_path, emot):
	#old_path could be   dir/dir/dir/file.mp3
	#file_name could be file.mp3
	#prefix could be dir/dir/dir/

	file_name = old_path.split('/')[-1]
	prefix = old_path[: old_path.index(file_name)]
	return prefix + emot + '/' + file_name



if __name__ == '__main__':


	choice = input('perform inference on one [i], or organize your musical library [o]?\n').strip()

	if choice == 'i':
		audio = input('Enter a youtube link or filepath to an mp3\n')

		t0 = time.time()

		if 'https' in audio:
			dl_audio_path = dl_audio(audio, None)
			specs = graph_spectrogram(dl_audio_path, 10, save = False)
			os.remove(dl_audio_path)
		else:
			specs = graph_spectrogram(audio, 10, save = False)


		perc = predict_song(specs)
		print_results(perc)
		print('took %f seconds to predict.' % (time.time() - t0))



	elif choice == 'o':
		directory = input('Enter filepath to directory of mp3 files')

		t0 = time.time()

		for emot in label_to_emot.values():
			if not os.path.exists(directory + '/' + emot):
				os.mkdir(directory + '/' + emot)

		audio_files = glob.glob(directory + '/*.mp3')

		for a in audio_files:
			specs = graph_spectrogram(a, 10, save = False)
			perc = predict_song(specs)
			predicted_emot = label_to_emot[np.argmax(perc)]

			new_path = emot_to_new_path(a, predicted_emot)

			os.rename(a, new_path)

		print('took %f seconds to organize.' % (time.time() - t0))
