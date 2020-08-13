#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g

    def __init__(self, e_char, e_word, kernel_size = 5, padding = 1):
        """
        @param e_char (int): length of character embedding vector
        @param e_word (int): length of word embedding vector
        """
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(e_char, e_word, kernel_size = kernel_size, padding = padding)

    def forward(self, input):
        """
        @param input: tensor of (batch_size, e_char, m_word)
        @returns output: tensor of (batch_size, e_word)
        """
        conv = self.conv(input) # (batch_size, e_word, _)
        output, _ = F.relu(conv).max(dim = 2) # (batch_size, e_word)
        return output

    ### END YOUR CODE

