from cv_stuff.parse_img import resize_crop, images_to_arrays, normalize_data, rand_rotate_and_crop
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import glob
from random import *
import pickle
import time


def split_data(all_data, all_labels, perc_train = 0.72, perc_val = 0.18, perc_test = 0.1):
	num_data = len(all_data)
	num_train = int(perc_train * num_data)
	num_val = int(perc_val * num_data)
	num_test = int(perc_test * num_data)

	curr = 0
	train_data = all_data[curr : num_train]
	train_labels = all_labels[curr : num_train]
	pickle.dump(train_data, open('processed_data/train_data.p', 'wb'))
	pickle.dump(train_labels, open('processed_data/train_labels.p', 'wb'))

	curr += num_train
	val_data = all_data[curr : curr + num_val]
	val_labels = labels[curr : curr + num_val]
	pickle.dump(val_data, open('processed_data/val_data.p', 'wb'))
	pickle.dump(val_labels, open('processed_data/val_labels.p', 'wb'))

	curr += num_val
	test_data = all_data[curr:]
	test_labels = all_labels[curr:]
	pickle.dump(test_data, open('processed_data/test_data.p', 'wb'))
	pickle.dump(test_labels, open('processed_data/test_labels.p', 'wb'))

def make_full_data(class_list):

	all_data = []
	all_labels = []

	i = 0
	while i < len(class_list):
		curr_class = class_list[i]
		data = pickle.load(open('processed_data/' + curr_class + '_data.p', 'rb'))
		all_data += list(data)
		all_labels += list(np.full(len(data), i))

	from sklearn.utils import shuffle
	all_data, all_labels = shuffle(all_data, all_labels)

	return all_data, all_labels


def make_data_per_class(class_list):

	for c in class_list:
		paths = glob.glob('org_data/' + c + '/*.jpg')
		imgs = rand_rotate_and_crop(paths)
		data = images_to_arrays(imgs)
		norm, _, _ = normalize_data(data)

		pickle.dump(norm, open('processed_data/' + c + '_data.p', 'wb'))


def view_data(start, end):

	data = pickle.load(open('processed_data/test_data.p', 'rb'))
	labels = pickle.load(open('processed_data/test_labels.p', 'rb'))
	print(len(data))
	for i in range(start, end):
		arr = np.reshape(data[i], (28, 28))
		print('label', labels[i])
		plt.gray()
		plt.imshow(arr)
		plt.show()

# start_time = time.time()
# make_data_per_class(class_list=['piano', 'guitar'])
# data, labels = make_full_data()
# split_data(data, labels)
# view_data(start = 280, end = 290)
# print('time to make data:', time.time() - start_time)
