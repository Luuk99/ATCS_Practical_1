# imports
import torch
import torch.nn as nn


# classifier class for the sentence representations to predictions
class Classifier(nn.Module):
    def __init__(self, input_dim=4*300):
        """Classifier used for the predictions
        Inputs:
            input_dim - Dimension of the input. Default is 4*300 (AWE embedding)
        """
        super().__init__()

        # initialize the classifier
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 3)
        )

    def forward(self, sentence_embeddings):
        """
        Inputs:
            sentence_embeddings - Tensor of sentence representations of shape [B, 4*300]
        Outputs:
            predictions - Tensor of predictions (entailment, neutral, contradiction) of shape [B, 3]
        """

        # pass sentence embeddings through model
        predictions = self.net(sentence_embeddings)

        # return the  predictions
        return predictions
