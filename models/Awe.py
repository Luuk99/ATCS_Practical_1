# imports
import torch
import torch.nn as nn
import numpy as np

# own imports
from dataset.LoadData import *


class AWEEncoder(nn.Module):
    def __init__(self):
        """Average Word Embedding Encoder

        Inputs:
            glove_embeddings - GloVe embeddings vocabulary
        """
        super().__init__()

        # create the glove embeddings
        self.glove_embeddings = load_glove()

    def forward(self, premises, hypothesis):
        """
        Inputs:
            premises - Input batch of sentence premises
            hypothesis - Input batch of sentence hypothesis
        Outputs:
            premise_embeddings - Tensor of premise representations
            hypothesis_embeddings - Tensor of hypothesis representations
        """

        # print the shapes
        print(premises.shape)
        print(hypothesis.shape)

        # return the premise and hypothesis embeddings
        return None, None

    @property
    def device(self):
        return next(self.parameters()).device
