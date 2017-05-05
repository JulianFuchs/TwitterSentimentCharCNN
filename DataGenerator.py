import tensorflow as tf
import numpy as np
import random

class DataGenerator():

    '''one data set has 1 billion tweets. We represent a tweet as 140 one hot vectors of size 75. So to store that
    we would need 10.5 TB... So we need read and convert to one hot for each batch separately'''
    def __init__(self, pos_path, neg_path, voc_path, max_seq, randomize):
        print('Reading vocabulary, reading and shuffling data...')

        self._pos_data = open(pos_path, 'r').read()
        self._neg_data = open(neg_path, 'r').read()
        self._voc_data = open(voc_path, 'r').read()

        self._voc_to_int = {}
        self._int_to_voc = {}

        self._unknown_chars = {}
        self._max_seq = max_seq

        self.generate_voc()

        self._pos_data = self._pos_data.split('\n')
        self._neg_data = self._neg_data.split('\n')

        if randomize:
            self._pos_data = random.sample(self._pos_data, len(self._pos_data))
            self._neg_data = random.sample(self._neg_data, len(self._neg_data))
            print('Shuffled data')

        #pos_converted = self.convert_data(self._pos_data)

        print('Vocabulary generated\n')

    '''reads vocabulary from voc.txt file'''
    def generate_voc(self):

        self._voc_to_int[' '] = 0
        self._int_to_voc[0] = ' '

        dic_index = 1

        '''exclude last element because it's \n'''
        for i in range(0, len(self._voc_data) - 1):
            char = self._voc_data[i]

            self._voc_to_int[char] = dic_index
            self._int_to_voc[dic_index] = char

            dic_index += 1

        print('Read vocabulary of size: ' + str(len(self._voc_to_int)))


    '''generates a batch from start to (start + batch_size) [not including last point] from data.
    The output is: [batch_size, voc_size, max_seq, 1]'''
    def generate_batch(self, data, start_ind, batch_size):

        batch = np.ndarray(shape=(batch_size, len(self._voc_to_int), self._max_seq, 1),
                               dtype=float)  # Creates a 2D array with uninitialized values

        batch_it = 0
        for tweet_it in range(start_ind, start_ind + batch_size):

            tweet = data[tweet_it]

            if tweet is None or len(tweet) == 0:
                print('Invalid tweet at index: ' + str(tweet_it))
            else:
                tweet_representation = self.generate_one_hot(tweet)

                batch[batch_it] = tweet_representation
                batch_it += 1

        return batch

    '''takes a tweet and converts it to a matrix: [voc_size, max_seq, 1]'''
    def generate_one_hot(self, tweet):

        tweet_representation = np.zeros(shape=(len(self._voc_to_int), self._max_seq, 1),
                               dtype=float)

        if len(tweet) > self._max_seq:
            tweet = tweet[0: 140]

        '''if char not known or tweet length is smaller 
        than max tweet length, take 0 vec which represents whitespace'''
        for char_it in range(0, len(tweet)):

            if tweet[char_it] in self._voc_to_int:
                int_rep = self._voc_to_int[tweet[char_it]]
                tweet_representation[int_rep][char_it][0] = 1
            else:
                self._unknown_chars[char_it] = 0

        return tweet_representation

    def find_longest_tweet(self):

        max_length = 0
        longest_tweet = ''

        for tweet in self._pos_data:
            if len(tweet) > max_length:
                max_length = len(tweet)
                longest_tweet = tweet

        for tweet in self._neg_data:
            if len(tweet) > max_length:
                max_length = len(tweet)
                longest_tweet = tweet

        print('Longest tweet was ' + str(max_length) + ' chars long. The tweet was:')
        print(longest_tweet)
        print()

if __name__ == '__main__':
    print('DataGenerator')
