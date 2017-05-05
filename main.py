import tensorflow as tf
import numpy as np
import DataGenerator
import CharCNNModel
import math
import time
import sys
import os

'''
Inspired by https://github.com/scharmchi/char-level-cnn-tf
'''
class Main:
    def __init__(self, pos_path, neg_path, epochs, num_threads):

        # load big data set
        # pos_path = '/home/theearlymiddleages/Datasets/twitter-datasets/train_pos_full.txt'
        # neg_path = '/home/theearlymiddleages/Datasets/twitter-datasets/train_neg_full.txt'

        # load small data set
        # pos_path = '/../../Datasets/twitter-datasets/train_pos.txt'
        # neg_path = '/../../Datasets/twitter-datasets/twitter-datasets/train_neg.txt'
        #current_directory = os.path.dirname(__file__)
        voc_path = str(os.path.dirname(os.path.realpath(__file__))) + '/voc.txt'

        self._data_generator = DataGenerator.DataGenerator(pos_path, neg_path, voc_path, 140, randomize=True)

        '''data points from [0, 5000] on both pos and neg tweets are test set. From [5000, 10000] test set. Rest is 
        training set'''
        self._test_set_offset = 0
        self._valid_set_offset = 5000
        self._validation_test_set_size = 5000

        self._batch_size = 128
        self._epochs = int(epochs)

        self._verbose_mode = False
        self._number_of_threads = int(num_threads)

        self._char_cnn_model = CharCNNModel.CharCNN(char_voc_size=len(self._data_generator._voc_data),
                                                    num_filters_per_layer=[256,256,256,256,256,256])

        config = tf.ConfigProto(intra_op_parallelism_threads=self._number_of_threads,
                                inter_op_parallelism_threads=self._number_of_threads)
        self._sess = tf.Session(config=config)

        init_op = tf.global_variables_initializer()
        self._sess.run(init_op)

        # self._data_generator.find_longest_tweet()

        #self.evaluate_on_validation()

        start = time.time()

        for e in range(0, self._epochs):
            print('Starting epoch ' + str(e + 1))

            self.run_epoch()

            print('After finishing epoch ' + str(e + 1))
            self.evaluate_on_validation()

            print('Number of unknown symbols encountered: ' + str(len(self._data_generator._unknown_chars)))
            print('')

        end = time.time()
        print()
        print('Entire program took ' + str((end - start) / 60) + ' minutes')

    def run_epoch(self):
        print('Training model on training set...')

        start_epoch = time.time()
        # take batch_size/2 positive examples

        index = self._valid_set_offset + self._validation_test_set_size
        half_batch_size = math.floor(self._batch_size / 2)

        # todo: actually include the final data points as well
        step = 0
        optimize_time_sum = 0

        max_length = max([len(self._data_generator._pos_data), len(self._data_generator._neg_data)])
        min_length = min([len(self._data_generator._pos_data), len(self._data_generator._neg_data)])

        while index + half_batch_size < min_length:

            '''[half_batch_size, voc_size, max_seq, 1]'''
            pos_examples = self._data_generator.generate_batch(self._data_generator._pos_data,
                                                               index, half_batch_size)

            neg_examples = self._data_generator.generate_batch(self._data_generator._neg_data,
                                                               index, half_batch_size)

            pos_labels = np.zeros(shape=(half_batch_size, 2), dtype=float)

            for i in range(0, half_batch_size):
                pos_labels[i, 1] = 1

            neg_labels = np.zeros(shape=(half_batch_size, 2), dtype=float)

            for i in range(0, half_batch_size):
                neg_labels[i, 0] = 1

            examples = np.concatenate((pos_examples, neg_examples), axis=0)

            labels = np.concatenate((pos_labels, neg_labels), axis=0)

            if step%100 == 0 and self._verbose_mode: #True: #
                acc = self._sess.run(self._char_cnn_model._accuracy, {self._char_cnn_model.input_x: examples,
                                                            self._char_cnn_model.input_y: labels,
                                                            self._char_cnn_model.dropout_keep_prob : 0.5})

                percentage = (step * half_batch_size) / (max_length - self._validation_test_set_size)

                print('Accuracy after ' + str(math.floor(percentage * 100)) + '%: '  + str(acc))
                # if step > 0:
                #     print('On average, an optimize call took: ' + str(optimize_time_sum / step) + ' seconds')

            '''model needs input:
            input_x: [batch_size, char_voc, max_seq, 1]
            input_y: [batch_size, classes]
            '''
            start_optimize = time.time()
            self._sess.run(self._char_cnn_model._optimize, {self._char_cnn_model.input_x: examples,
                                                            self._char_cnn_model.input_y: labels,
                                                            self._char_cnn_model.dropout_keep_prob: 0.5})
            end_optimize = time.time()

            optimize_time_sum += end_optimize-start_optimize

            index += half_batch_size

            step += 1

        end_epoch = time.time()
        print('Finished training one epoch')
        print('Training took ' + str((end_epoch - start_epoch) / 60) + ' minutes')
        print('On average, an optimize call took: ' + str(optimize_time_sum/step) + ' seconds\n')

    def evaluate_on_validation(self):

        print('Evaluating model on validation set...')
        start = time.time()

        half_batch_size = math.floor(self._batch_size / 2)

        accuracies = []

        step = 0
        index = self._valid_set_offset

        while index < self._valid_set_offset + self._validation_test_set_size:

            pos_examples = self._data_generator.generate_batch(self._data_generator._pos_data,
                                                               index, half_batch_size)

            neg_examples = self._data_generator.generate_batch(self._data_generator._neg_data,
                                                               index, half_batch_size)

            pos_labels = np.zeros(shape=(half_batch_size, 2), dtype=float)

            for i in range(0, half_batch_size):
                pos_labels[i, 1] = 1

            neg_labels = np.zeros(shape=(half_batch_size, 2), dtype=float)

            for i in range(0, half_batch_size):
                neg_labels[i, 0] = 1

            examples = np.concatenate((pos_examples, neg_examples), axis=0)

            labels = np.concatenate((pos_labels, neg_labels), axis=0)


            acc = self._sess.run(self._char_cnn_model._accuracy, {self._char_cnn_model.input_x: examples,
                                                                  self._char_cnn_model.input_y: labels,
                                                                  self._char_cnn_model.dropout_keep_prob: 0.5})

            accuracies.append(acc)

            if step % 100 == 0 and self._verbose_mode:  # True: #
                percentage = (step * half_batch_size) / self._validation_test_set_size
                print('Accuracy after ' + str(math.floor(percentage*100)) + '%: ' + str(acc))

            index += half_batch_size
            step += 1

        end = time.time()
        total_accuracy = sum(accuracies)/len(accuracies)

        print('Evaluating validation set took ' + str((end - start)/60) + ' minutes')
        print('Accuracy over entire validation set: ' + str(math.floor(100*total_accuracy)) + '%')


if __name__ == '__main__':

    if len(sys.argv) != 5:
        print('To run model, input: pos_path, neg_path, epochs, num_threads')
    else:
        print('Starting Tweet Sentiment char based CNN analysis \n')
        Main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])


