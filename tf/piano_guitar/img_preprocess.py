from cv_stuff.parse_img import resize_crop, images_to_arrays, normalize_data
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import glob
from random import *
import pickle



def preprocess(file_paths_list, rots_per_img = 10, crops_per_rot = 5):

    step = 0
    processed_images = []

    for path in file_paths_list:
        img = Image.open(path)

        for _ in range(rots_per_img):

            rot = int(gauss(0, 1.4) * 13)
            rotated = img.rotate(rot)

            f = randint(0, 1)
            if f == 1:
                rotated = rotated.transpose(Image.FLIP_LEFT_RIGHT)

            for _ in range(crops_per_rot):
                i = resize_crop(img = rotated, crop_type='random')
                processed_images.append(i)
                step += 1

        print('.')
    return processed_images

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
    val_data = all_data[curr : num_val]
    val_labels = labels[curr : num_val]
    pickle.dump(val_data, open('processed_data/val_data.p', 'wb'))
    pickle.dump(val_labels, open('processed_data/val_labels.p', 'wb'))

    curr += num_val
    test_data = all_data[curr:]
    test_labels = all_labels[curr:]
    pickle.dump(test_data, open('processed_data/test_data.p', 'wb'))
    pickle.dump(test_labels, open('processed_data/test_labels.p', 'wb'))

def make_full_data():

    piano_norm = pickle.load(open('processed_data/piano_data.p', 'rb'))
    guitar_norm = pickle.load(open('processed_data/guitar_data.p', 'rb'))
    print('piano', len(piano_norm))
    print('guitar', len(guitar_norm))

    all_data = list(piano_norm) + list(guitar_norm)

    all_labels = list(np.full(len(piano_norm), 0)) + list(np.full(len(guitar_norm), 1))

    from sklearn.utils import shuffle
    all_data, all_labels = shuffle(all_data, all_labels)

    return all_data, all_labels


    print('Success! :)')


def make_data_per_class():
    piano_paths = glob.glob('org_data/piano/*.jpg')
    guitar_paths = glob.glob('org_data/guitar/*.jpg')

    piano_imgs = preprocess(piano_paths)
    guitar_imgs = preprocess(guitar_paths)

    piano_data = images_to_arrays(piano_imgs)
    guitar_data = images_to_arrays(guitar_imgs)

    piano_norm, mean, sd = normalize_data(piano_data)
    guitar_norm, mean, sd = normalize_data(guitar_data)

    pickle.dump(piano_norm, open('processed_data/piano_data.p', 'wb'))
    pickle.dump(guitar_norm, open('processed_data/guitar_data.p', 'wb'))

def view_data(start = 50, end = 60):

    data = pickle.load(open('processed_data/test_data.p', 'rb'))

    print(len(data))

    for i in range(start, end):
        arr = np.reshape(data[i], (28, 28))
        plt.gray()
        plt.imshow(arr)
        plt.show()

# data, labels = make_full_data()
# split_data(data, labels)
view_data()
