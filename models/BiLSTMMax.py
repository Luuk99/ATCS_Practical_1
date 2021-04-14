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
        hidden_states, _ = self.lstm(packed_sentences, (self.hidden_state, self.cell_state))

        # pad the output hidden states
        hidden_states = nn.utils.rnn.pad_packed_sequence(hidden_states, batch_first=True)

        # unsort the embeddings
        unsorted_indices = torch.argsort(sorted_indices)
        sentence_representations = torch.index_select(hidden_states[0], dim=0, index=unsorted_indices)

        # apply max pooling
        max_sentences = []
        for index, sentence in enumerate(sentence_representations):
            # cut-off the sentence
            sentence = sentence[:(sentence_lengths[index])]

            # max the sentence
            sentence = torch.max(sentence, dim=0)[0]

            # add to the list
            max_sentences.append(sentence)

        # stack the tensors
        sentence_representations = torch.stack(max_sentences, dim=0)

        # return the sentence representations
        return sentence_representations

    @property
    def device(self):
        return next(self.parameters()).device
