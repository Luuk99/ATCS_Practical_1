# imports
import torch
import torch.nn as nn
import numpy as np

# own imports
from dataset.LoadData import load_glove


class AWEEncoder(nn.Module):
    def __init__(self):
        """Average Word Embedding Encoder
        """
        super().__init__()

        # create the glove embeddings
        glove_embeddings = load_glove()
        glove_embeddings = glove_embeddings.vectors

        # create an embedding layer for the glove embeddings
        self.glove_embeddings = nn.Embedding.from_pretrained(glove_embeddings)

    def forward(self, premises, hypothesis):
        """
        Inputs:
            premises - Input batch of sentence premises
            hypothesis - Input batch of sentence hypothesis
        Outputs:
            sentence_representations - Tensor of sentence representations of shape [B, 4*300]
        """

        # pass premises and hypothesis through the embeddings
        premises = self.glove_embeddings(premises)
        hypothesis = self.glove_embeddings(hypothesis)

        # mean the embeddings for AWE representation
        premises = torch.mean(premises, dim=1)
        hypothesis = torch.mean(hypothesis, dim=1)

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
