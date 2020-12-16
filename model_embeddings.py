#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn


# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        # Test calls use this, my this name
        self.embed_size = embed_size
        self.embed_word_size = embed_size
        self.vocab = vocab # vocab for a particular language
        self.dropout_prob = 0.3
        self.embed_char_size = 50
        self.max_vocab_tokens_in_word = 21
        self.embed_layer = nn.Embedding(num_embeddings=len(self.vocab.char2id), embedding_dim=self.embed_char_size, padding_idx=vocab.char2id['<pad>'])
        self.cnn_module = CNN(num_input_channels=self.embed_char_size, num_out_channels=self.embed_word_size, input_embed_size=self.max_vocab_tokens_in_word)
        self.highway_module = Highway(word_embedding_size=self.embed_word_size)
        self.dropout = nn.Dropout(p=self.dropout_prob)
        ### END YOUR CODE

    def forward(self, input_tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input_tensor: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        X_char_embeded = self.embed_layer(input_tensor) # of shape (sentence_length, batch_size, max_word_length, embed_char_size)
        X_char_embeded = X_char_embeded.permute(0, 1, 3, 2) # of shape (sentence_length, batch_size, embed_char_size, max_word_length)
        X_char_embeded_sizes = list(X_char_embeded.size())
        X_char_embeded_for_cnn_input = X_char_embeded.contiguous().view(X_char_embeded_sizes[0]*X_char_embeded_sizes[1], X_char_embeded_sizes[2], X_char_embeded_sizes[3])
        X_out = self.cnn_module.forward(X_char_embeded_for_cnn_input) # output of shape (sentence_length*batch_size, embed_word_size)
        X_out = self.highway_module.forward(X_out)
        X_out = self.dropout(X_out) # output of shape (sentence_length*batch_size, embed_word_size)
        X_out = X_out.contiguous().view(X_char_embeded_sizes[0], X_char_embeded_sizes[1], -1)
        # assert list(X_out.size()) == [X_char_embeded_sizes[0], X_char_embeded_sizes[1], self.embed_word_size]
        return(X_out)
        ### END YOUR CODE
