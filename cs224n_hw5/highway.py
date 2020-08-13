#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f

    def __init__(self, e_word):
        """
        @param e_word (int): length of word embedding vector
        """
        super(Highway, self).__init__()
        self.proj = nn.Linear(e_word, e_word)
        self.gate = nn.Linear(e_word, e_word)

    def forward(self, input):
        """
        @param input: tensor of (batch_size, e_word)
        @returns output: tensor of (batch_size, e_word)
        """
        proj = F.relu(self.proj(input))
        gate = self.gate(input).sigmoid()
        output = gate * proj + (1 - gate) * input
        return output

    ### END YOUR CODE

