# imports
import torch
import torch.nn as nn
import numpy as np


class Classifier(nn.Module):
    def __init__(self):
        """Classifier used for the predictions
        """
        super().__init__()

        # initialize the classifier
        self.net = nn.Sequential(
            nn.Linear(3*300, 512),
            nn.Linear(512, 3)
        )

        # initialize the softmax
        self.softmax = nn.Softmax()

    def forward(self, sentence_embeddings):
        """
        Inputs:
            sentence_embeddings - Tensor of sentence representations
        Outputs:
            predictions - Tensor of predictions (entailment, neutral, contradiction)
        """

        # pass sentence embeddings through model
        predictions = self.net(sentence_embeddings)

        # softmax the predctions
        predictions = self.softmax(predictions)

        # return the  predictions
        return predictions
