# imports
import torch
import torch.nn as nn


# classifier class for the sentence representations to predictions
class Classifier(nn.Module):
    def __init__(self):
        """Classifier used for the predictions
        """
        super().__init__()

        # initialize the classifier
        self.net = nn.Sequential(
            nn.Linear(4*300, 512),
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

        # DEBUG
        # append 0's before the predictions because labels range from 1-3
        predictions = torch.cat([torch.zeros((sentence_embeddings.shape[0], 1), device=predictions.device), predictions], dim=-1)

        # return the  predictions
        return predictions
