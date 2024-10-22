import pickle
import random
from sklearn.utils import shuffle as skshuffle
import numpy as np
from utils.parse_img import normalize_data

class DataUtil:

    def __init__(self, data_dir, batch_size, num_epochs, normalize = True, supervised=True):


        if not supervised:
            self.images_train = list(pickle.load(open(data_dir + '/train_data.p', 'rb')))
            random.shuffle(self.images_train)

        else:
            self.images_train = list(pickle.load(open(data_dir + '/train_data.p', 'rb')))
            self.labels_train = list(pickle.load(open(data_dir + '/train_labels.p', 'rb')))

            images_val = list(pickle.load(open(data_dir + '/val_data.p', 'rb')))

            if normalize:
                images_val = normalize_data(images_val)

            self.images_val = images_val
            self.labels_val = list(pickle.load(open(data_dir + '/val_labels.p', 'rb')))



        self.batch_size = batch_size
        self.curr_data_num = 0
        self.global_num = 0

        self.curr_epoch = 0
        self.num_epochs = num_epochs

        self.supervised = supervised


    # def normalize_data(my_data, return_mean_and_std = False):
    #
    #     '''
    #     Args:
    #         2D array with arr storing each image, and arr[i] storing pixels of image i
    #     Returns:
    #         normalized my_data, mean of my_data, standard deviation of my_data
    #     '''
    #     m = np.mean(my_data, axis = 0)
    #     std = np.std(my_data, axis = 0)
    #
    #     my_data -= m
    #     my_data /= (std + 1e-8)
    #
    #     if return_mean_and_std:
    #         return my_data, m, std
    #     else:
    #         return my_data


    def get_next_batch(self):
        '''
        Returns:
            Next training batch, None if finished all epochs
        '''


        if self.supervised:
            if self.curr_epoch >= self.num_epochs:
                return None, None
            else:
                return self.get_next_batch_with_labels()

        else:
            if self.curr_epoch >= self.num_epochs:
                return None
            else:
                return self.get_next_batch_without_labels()



    def get_next_batch_without_labels(self):

        '''
        Gets the next batch in training data. WITHOUT LABELS
        @param None
        @return The next normalized training DATA BATCH
        '''

        img_batch = []

        for _ in range(self.batch_size):

            img_batch.append(self.images_train[self.curr_data_num])

            self.curr_data_num += 1
            self.global_num += 1

            if self.curr_data_num > len(self.images_train) - 1:

                print('FINISHED EPOCH', self.curr_epoch + 1)
                self.curr_epoch += 1
                self.curr_data_num = 0
                random.shuffle(self.images_train)


        img_batch = normalize_data(img_batch)

        return img_batch



    def get_next_batch_with_labels(self):
        '''
        Gets the next batch in training data. WITH LABELS
        @param None
        @return The next normalized training batch
        '''

        img_batch = []
        labels_batch = []

        for _ in range(self.batch_size):

            img_batch.append(self.images_train[self.curr_data_num])
            labels_batch.append(self.labels_train[self.curr_data_num])

            self.curr_data_num += 1
            self.global_num += 1

            if self.curr_data_num > len(self.images_train) - 1:

                print('FINISHED EPOCH', self.curr_epoch + 1)
                self.curr_epoch += 1
                self.curr_data_num = 0
                self.images_train, self.labels_train = skshuffle(self.images_train, self.labels_train)


        img_batch = normalize_data(img_batch)

        return img_batch, labels_batch




def split_data(save_folder, all_data, all_labels, perc_train = 0.72, perc_val = 0.18, perc_test = 0.1):
    num_data = len(all_data)
    num_train = int(perc_train * num_data)
    num_val = int(perc_val * num_data)
    num_test = int(perc_test * num_data)


    curr = 0
    train_data = all_data[curr : num_train]
    train_labels = all_labels[curr : num_train]
    pickle.dump(train_data, open(save_folder + '/train_data.p', 'wb'))
    pickle.dump(train_labels, open(save_folder + '/train_labels.p', 'wb'))

    curr += num_train
    val_data = all_data[curr : curr + num_val]
    val_labels = all_labels[curr : curr + num_val]
    pickle.dump(val_data, open(save_folder + '/val_data.p', 'wb'))
    pickle.dump(val_labels, open(save_folder + '/val_labels.p', 'wb'))

    curr += num_val
    test_data = all_data[curr:]
    test_labels = all_labels[curr:]
    pickle.dump(test_data, open(save_folder + '/test_data.p', 'wb'))
    pickle.dump(test_labels, open(save_folder + '/test_labels.p', 'wb'))
