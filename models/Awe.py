# imports
import torch
import torch.nn as nn
import numpy as np


# AWE encoder class for creating the sentence representations
class AWEEncoder(nn.Module):
    def __init__(self):
        """Average Word Embedding Encoder
        """
        super().__init__()

    def forward(self, sentences, sentence_lengths):
        """
        Inputs:
            sentences - Input batch of sentences
            sentence_lengths - List of unpadded sentence lengths
        Outputs:
            sentence_representations - Tensor of sentence representations of shape [B, 300]
        """

        # remove the paddings and mean the representations
        sentence_representations = []
        for index, sentence in enumerate(sentences):
            # cut-off the sentence
            sentence = sentence[:(sentence_lengths[index])]

            # mean the sentence
            sentence = torch.mean(sentence, dim=0)

            # add to the list
            sentence_representations.append(sentence)

        # stack the tensors
        sentence_representations = torch.stack(sentence_representations, dim=0)

        # return the sentence representations
        return sentence_representations

    @property
    def device(self):
        return next(self.parameters()).device
