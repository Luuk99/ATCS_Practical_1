# imports
import torch
import torch.nn as nn
import numpy as np


# Unidirectional LSTM encoder class for creating the sentence representations
class UniLSTM(nn.Module):
    def __init__(self, batch_size=64):
        """Unidirectional LSTM Encoder
        Inputs:
            batch_size - Size of the batches. Default is 64
        """
        super().__init__()

        # save the batch size
        self.batch_size = batch_size

        # create the LSTM
        self.lstm = nn.LSTM(input_size=300, hidden_size=2048, num_layers=1,
                            batch_first=True, bidirectional=False)

    def forward(self, premises, lengths_premises, hypothesis, lengths_hypothesis):
        """
        Inputs:
            premises - Input batch of sentence premises
            lengths_premises - List of unpadded premise lengths
            hypothesis - Input batch of sentence hypothesis
            lengths_hypothesis - List of unpadded hypothesis lengths
        Outputs:
            sentence_representations - Tensor of sentence representations of shape [B, 4*2048]
        """

        # initialize the hidden state and cell state
        self.hidden_state = torch.zeros((1, premises.shape[0], 2048), dtype=torch.float, device=self.device)
        self.cell_state = torch.zeros((1, premises.shape[0], 2048), dtype=torch.float, device=self.device)

        # sort the embeddings on sentence length
        sorted_lengths_premises, sorted_indices_premises = torch.sort(lengths_premises, dim=0, descending=True)
        sorted_lengths_hypothesis, sorted_indices_hypothesis = torch.sort(lengths_hypothesis, dim=0, descending=True)
        sorted_premises = torch.index_select(premises, dim=0, index=sorted_indices_premises)
        sorted_hypothesis = torch.index_select(hypothesis, dim=0, index=sorted_indices_hypothesis)

        # pack the embeddings
        packed_premises = nn.utils.rnn.pack_padded_sequence(sorted_premises,sorted_lengths_premises, batch_first=True)
        packed_hypothesis = nn.utils.rnn.pack_padded_sequence(sorted_hypothesis, sorted_lengths_hypothesis, batch_first=True)

        # run through the model
        _, premises_hidden_states = self.lstm(packed_premises, (self.hidden_state, self.cell_state))
        _, hypothesis_hidden_states = self.lstm(packed_hypothesis, (self.hidden_state, self.cell_state))
        premises_hidden_states = premises_hidden_states[0].squeeze(dim=0)
        hypothesis_hidden_states = hypothesis_hidden_states[0].squeeze(dim=0)

        # unsort the embeddings
        unsorted_indices_premises = torch.argsort(sorted_indices_premises)
        unsorted_indices_hypothesis = torch.argsort(sorted_indices_hypothesis)
        premises = torch.index_select(premises_hidden_states, dim=0, index=unsorted_indices_premises)
        hypothesis = torch.index_select(hypothesis_hidden_states, dim=0, index=unsorted_indices_hypothesis)

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
