import tensorflow as tf
import os
from generate_data import dl_audio
from make_spectrogram import graph_spectrogram
from PIL import Image


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
		print(label_to_emot[label])



if __name__ == '__main__':

	inference(Image.open('processed_data/angry/0XoyDqFy5pU_2_3.jpg'))

	quit()

	choice = input('perform inference on one [i], or organize your musical library [o]?').strip()

	if choice == 'i':
		audio = input('Enter a youtube link or filepath to an mp3 filepath')

		if 'https' in audio:
			audio = dl_audio(audio, None)

		specs = graph_spectrogram(audio, 10, save = False)


	elif choice == 'o':
		directory = input('Enter filepath to directory of mp3 files')





