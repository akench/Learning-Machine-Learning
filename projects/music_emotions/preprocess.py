import utils.parse_img as parse_img
import glob
from PIL import Image
from random import *

def preprocess_per_emot(emot, num_stretch = 3, num_salt = 2):

    paths = glob.glob('gen_specs/' + emot + '/*')
    # paths = ['gen_specs/motivational/tGh4FcZKekA0.jpg']
    processed = []

    for p in paths:
        img = Image.open(p).convert('L')
        img = img.resize((128, 128))
        print('.', end='', flush=True)

        for _ in range(num_stretch):
            r = uniform(0.9, 1.1)
            stretched = parse_img.stretch_img(img, stretch_type = 'w', factor = r)
            stretched = parse_img.resize_crop(stretched, size=128)

            for _ in range(num_salt):
                processed.append(parse_img.salt_and_pepper(stretched, 0.001))

    for im in processed:
        im.save('testdir/' + str(randint(0, 100000000000)) + '.jpg')

    print('SAVED')


preprocess_per_emot('motivational')
