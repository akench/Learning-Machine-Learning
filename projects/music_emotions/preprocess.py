import utils.parse_img as parse_img
import glob
from PIL import Image
from random import *
import pickle

def preprocess_per_emot(emot, num_stretch = 4, num_salt = 2):

    paths = glob.glob('gen_specs/' + emot + '/*')
    processed = []
    file_names = []

    for p in paths:

        vid_id = p.split('/')[-1].split('.')[0]
        i = 0

        img = Image.open(p).convert('L')
        img = img.resize((128, 128))
        print('.', end='', flush=True)

        for _ in range(num_stretch):
            r = uniform(0.9, 1.1)
            stretched = parse_img.stretch_img(img, stretch_type = 'w', factor = r)
            stretched = parse_img.resize_crop(stretched, size=128)

            for _ in range(num_salt):
                processed.append(parse_img.salt_and_pepper(stretched, 0.001))
                file_names.append(vid_id + str(i))
                i += 1


    for im, name in zip(processed, file_names):
        im.save('processed_data/' + emot + '/' + name + '.jpg')

    print('SAVED')


def preprocess_all():
    dirs = glob.glob('gen_specs/*')
    emots = [d.split('/')[1] for d in dirs]

    for e in emots:
        preprocess_per_emot(e)


def make_data():

    emots = glob.glob('processed_data/*/')
    all_data = []
    all_labels = []

    curr_label = 0
    for e in emots:
        print(e)
        image_paths = glob.glob(e + '/*')

        for path in image_paths:
            img = Image.open(path)
            arr = parse_img.images_to_arrays([img])
            all_data.append(arr)
            all_labels.append(curr_label)

        curr_label += 1

    from sklearn.utils import shuffle as skshuffle
    all_data, all_labels = skshuffle(all_data, all_labels)



    from utils.split_data import split_data
    split_data('processed_data', all_data, all_labels)

make_data()
