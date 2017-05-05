# TwitterSentimentCharCNN
Character based CNN for Twitter Sentiment Classification

Takes the Twitter data from the Computational Intelligence Lab at ETH from the spring semester 2016 (Project in Exercise 6) and performs a Text Sentiment Classification using a Character based Convolutional Network (CNN).

The CNN used is 9 layers deep with 6 convolutional layers and 3 fully connected layers, heavily inspired by the CNN introduced in the paper 'Character-level Convolutional Networks for Text Classification' by Xiang Zhang, Junbo Zhao, Yann LeCun. https://arxiv.org/abs/1509.01626

As a base code, the code written by https://github.com/scharmchi was used found under https://github.com/scharmchi/char-level-cnn-tf. The code was modified however: 

 - Padding was introduced (a padding of 'SAME' instead of 'VALID' in tensorflow)
 - Since the maximal twitter length is 140 and the original paper has a sequence length of 1014 in mind, we only use 2 pooling steps instead of 3 as in the original paper, since otherwise the vectors would get too small
 - You can manually set the number of filters for each convolutional layer

test
