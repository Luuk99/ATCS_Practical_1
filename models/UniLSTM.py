# imports
import torch
import torch.nn as nn
import numpy as np


# Unidirectional LSTM encoder class for creating the sentence representations
class UniLSTM(nn.Module):
    def __init__(self):
        """Unidirectional LSTM Encoder
        """
        super().__init__()

        # create the LSTM
        self.lstm = nn.LSTM(input_size=300, hidden_size=2048, num_layers=1,
                            batch_first=True, bidirectional=False)

    def forward(self, sentences, sentence_lengths):
        """
        Inputs:
            sentences - Input batch of sentences
            sentence_lengths - List of unpadded sentence lengths
        Outputs:
            sentence_representations - Tensor of sentence representations of shape [B, 2048]
        """

        # initialize the hidden state and cell state
        self.hidden_state = torch.zeros((1, sentences.shape[0], 2048), dtype=torch.float, device=self.device)
        self.cell_state = torch.zeros((1, sentences.shape[0], 2048), dtype=torch.float, device=self.device)

        # sort the embeddings on sentence length
        sorted_lengths, sorted_indices= torch.sort(sentence_lengths, dim=0, descending=True)
        sorted_sentences = torch.index_select(sentences, dim=0, index=sorted_indices)

        # pack the embeddings
        packed_sentences = nn.utils.rnn.pack_padded_sequence(sorted_sentences, sorted_lengths, batch_first=True)

        # DEBUG
        print(packed_sentences)

        # run through the model
        _, (hidden_states, _) = self.lstm(packed_sentences, (self.hidden_state, self.cell_state))
        hidden_states = hidden_states.squeeze(dim=0)

        # unsort the embeddings
        unsorted_indices = torch.argsort(sorted_indices)
        sentence_representations = torch.index_select(hidden_states, dim=0, index=unsorted_indices)

        # return the sentence representations
        return sentence_representations

    @property
    def device(self):
        return next(self.parameters()).device
