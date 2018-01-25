import pickle


class DataUtil:

    def __init__(self, data_dir, batch_size, num_epochs, normalize = False):
        self.images_train = list(pickle.load(open(data_dir + '/train_data.p', 'rb')))
        self.labels_train = list(pickle.load(open(data_dir + '/train_labels.p', 'rb')))

        images_val = list(pickle.load(open(data_dir + '/val_data.p', 'rb')))

        if normalize:
            images_val, _, _ = normalize_data(images_val)

        self.images_val = images_val
        self.labels_val = list(pickle.load(open(data_dir + '/val_labels.p', 'rb')))

        self.batch_size = batch_size
        self.curr_data_num = 0
        self.global_num = 0

        self.curr_epoch = 0
        self.num_epochs = num_epochs


    def get_next_batch(self):
        '''
        Gets the next batch in training data.
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

                print('FINISHED EPOCH', self.curr_epoch)
                self.curr_epoch += 1
                self.curr_data_num = 0
                self.images_train, self.labels_train = shuffle(self.images_train, self.labels_train)


        # img_batch, _, _ = normalize_data(img_batch)

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
