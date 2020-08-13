#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch.nn.functional as F

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab, char_embed_size = 50, dropout_prob = 0.3):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ### YOUR CODE HERE for part 1h

        self.word_embed_size = word_embed_size # for the autograder
        self.e_word = word_embed_size
        self.e_char = char_embed_size
        self.dropout_prob = dropout_prob
        self.embedding = nn.Embedding(len(vocab.char2id), self.e_char, vocab.char_pad)
        self.cnn = CNN(self.e_char, self.e_word)
        self.highway = Highway(self.e_word)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h

        sentence_length, batch_size, max_word_length = input.shape
        x_reshaped = self.embedding(input)                             # (sentence_length, batch_size, max_word_length, self.e_char)
        x_reshaped = x_reshaped.view(-1, max_word_length, self.e_char) # (sentence_length * batch_size, max_word_length, self.e_char)
        x_reshaped = x_reshaped.transpose(1, 2)                        # (sentence_length * batch_size, self.e_char, max_word_length)
        x_conv_out = self.cnn(x_reshaped)                              # (sentence_length * batch_size, self.e_word)
        x_highway = self.highway(x_conv_out)                           # (sentence_length * batch_size, self.e_word)
        x_highway = x_highway.view(-1, batch_size, self.e_word)        # (sentence_length, batch_size, self.e_word)
        output = F.dropout(x_highway, self.dropout_prob)
        return output

        ### END YOUR CODE

