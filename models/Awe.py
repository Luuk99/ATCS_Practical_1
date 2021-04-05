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

    def forward(self, premises, lengths_premises, hypothesis, lengths_hypothesis):
        """
        Inputs:
            premises - Input batch of sentence premises
            lengths_premises - List of unpadded premise lengths
            hypothesis - Input batch of sentence hypothesis
            lengths_hypothesis - List of unpadded hypothesis lengths
        Outputs:
            sentence_representations - Tensor of sentence representations of shape [B, 4*300]
        """

        # remove the paddings and mean the representations
        average_premises = []
        for index, sentence in enumerate(premises):
            # cut-off the sentence
            sentence = sentence[:(lengths_premises[index])]

            # mean the sentence
            sentence = torch.mean(sentence, dim=0)

            # add to the list
            average_premises.append(sentence)
        average_hypothesis = []
        for index, sentence in enumerate(hypothesis):
            # cut-off the sentence
            sentence = sentence[:(lengths_hypothesis[index])]

            # mean the sentence
            sentence = torch.mean(sentence, dim=0)

            # add to the list
            average_hypothesis.append(sentence)

        # stack the tensors
        premises = torch.stack(average_premises, dim=0)
        hypothesis = torch.stack(average_hypothesis, dim=0)

        # calculate the difference and multiplication
        difference = torch.abs(premises - hypothesis)
        multiplication = premises * hypothesis

        # create the sentence representations
        sentence_representations = torch.cat([premises, hypothesis, difference, multiplication], dim=1)

        # return the sentence representations
        return sentence_representations

    @property
    def device(self):
        return next(self.parameters()).device
