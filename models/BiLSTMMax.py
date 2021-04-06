# imports
import torch
import torch.nn as nn
import numpy as np


# Bidirectional LSTM encoder with max pooling class for creating the sentence representations
class BiLSTMMax(nn.Module):
    def __init__(self):
        """Bidirectional LSTM Encoder
        """
        super().__init__()

        # create the LSTM
        self.lstm = nn.LSTM(input_size=300, hidden_size=4096, num_layers=1,
                            batch_first=True, bidirectional=True)

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
        self.hidden_state = torch.zeros((2, premises.shape[0], 4096), dtype=torch.float, device=self.device)
        self.cell_state = torch.zeros((2, premises.shape[0], 4096), dtype=torch.float, device=self.device)

        # sort the embeddings on sentence length
        sorted_lengths_premises, sorted_indices_premises = torch.sort(lengths_premises, dim=0, descending=True)
        sorted_lengths_hypothesis, sorted_indices_hypothesis = torch.sort(lengths_hypothesis, dim=0, descending=True)
        sorted_premises = torch.index_select(premises, dim=0, index=sorted_indices_premises)
        sorted_hypothesis = torch.index_select(hypothesis, dim=0, index=sorted_indices_hypothesis)

        # pack the embeddings
        packed_premises = nn.utils.rnn.pack_padded_sequence(sorted_premises,sorted_lengths_premises, batch_first=True)
        packed_hypothesis = nn.utils.rnn.pack_padded_sequence(sorted_hypothesis, sorted_lengths_hypothesis, batch_first=True)

        # run through the model
        premises_hidden_states, _ = self.lstm(packed_premises, (self.hidden_state, self.cell_state))
        hypothesis_hidden_states, _ = self.lstm(packed_hypothesis, (self.hidden_state, self.cell_state))

        # pad the output hidden states
        premises_hidden_states = nn.utils.rnn.pad_packed_sequence(premises_hidden_states, batch_first=True)
        hypothesis_hidden_states = nn.utils.rnn.pad_packed_sequence(hypothesis_hidden_states, batch_first=True)

        # unsort the embeddings
        unsorted_indices_premises = torch.argsort(sorted_indices_premises)
        unsorted_indices_hypothesis = torch.argsort(sorted_indices_hypothesis)
        premises = torch.index_select(premises_hidden_states[0], dim=0, index=unsorted_indices_premises)
        hypothesis = torch.index_select(hypothesis_hidden_states[0], dim=0, index=unsorted_indices_hypothesis)

        # apply max pooling
        max_premises = []
        for index, sentence in enumerate(premises):
            # cut-off the sentence
            sentence = sentence[:(lengths_premises[index])]

            # max the sentence
            sentence = torch.max(sentence, dim=0)[0]

            # add to the list
            max_premises.append(sentence)
        max_hypothesis = []
        for index, sentence in enumerate(hypothesis):
            # cut-off the sentence
            sentence = sentence[:(lengths_hypothesis[index])]

            # max the sentence
            sentence = torch.max(sentence, dim=0)[0]

            # add to the list
            max_hypothesis.append(sentence)

        # stack the tensors
        premises = torch.stack(max_premises, dim=0)
        hypothesis = torch.stack(max_hypothesis, dim=0)

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
