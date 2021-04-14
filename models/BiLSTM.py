# imports
import torch
import torch.nn as nn
import numpy as np


# Bidirectional LSTM encoder class for creating the sentence representations
class BiLSTM(nn.Module):
    def __init__(self):
        """Bidirectional LSTM Encoder
        """
        super().__init__()

        # create the LSTM
        self.lstm = nn.LSTM(input_size=300, hidden_size=2048, num_layers=1,
                            batch_first=True, bidirectional=True)

    def forward(self, sentences, sentence_lengths):
        """
        Inputs:
            sentences - Input batch of sentences
            sentence_lengths - List of unpadded sentence lengths
        Outputs:
            sentence_representations - Tensor of sentence representations of shape [B, 2*2048]
        """

        # initialize the hidden state and cell state
        self.hidden_state = torch.zeros((2, sentences.shape[0], 2048), dtype=torch.float, device=self.device)
        self.cell_state = torch.zeros((2, sentences.shape[0], 2048), dtype=torch.float, device=self.device)

        # sort the embeddings on sentence length
        sorted_lengths, sorted_indices= torch.sort(sentence_lengths, dim=0, descending=True)
        sorted_sentences = torch.index_select(sentences, dim=0, index=sorted_indices)

        # pack the embeddings
        packed_sentences = nn.utils.rnn.pack_padded_sequence(sorted_sentences, sorted_lengths, batch_first=True)

        # run through the model
        _, (hidden_states, _) = self.lstm(packed_sentences, (self.hidden_state, self.cell_state))

        # get the forward and backward hidden states
        forward_state = hidden_states[0]
        backward_state = hidden_states[1]

        # check the dimensionality of the states
        if (len(list(forward_state.shape)) > 2):
            forward_state = forward_state.squeeze(dim=0)
            backward_state = backward_state.squeeze(dim=0)

        # concat the hidden states of the directions
        hidden_states = torch.cat([forward_state, backward_state], dim=1)

        # unsort the embeddings
        unsorted_indices = torch.argsort(sorted_indices)
        sentence_representations = torch.index_select(hidden_states, dim=0, index=unsorted_indices)

        # return the sentence representations
        return sentence_representations

    @property
    def device(self):
        return next(self.parameters()).device
