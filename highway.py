#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1d
import torch
import torch.nn as nn
import torch.nn.utils

class Highway(nn.Module):
    def __init__(self, word_embedding_size):
        super(Highway, self).__init__()
        self.word_embedding_size = word_embedding_size
        self.projection_linear = nn.Linear(in_features=self.word_embedding_size, out_features=self.word_embedding_size, bias=True)
        self.relu = nn.ReLU()
        self.gate_linear = nn.Linear(in_features=self.word_embedding_size, out_features=self.word_embedding_size, bias=True)

    def forward(self, X_conv_out) -> torch.Tensor:
        # X_conv_out in batched form should be a (batch_size x * x word_embedding_size
        # The output dim also stays the same in dimension
        X_proj = self.relu(self.projection_linear(X_conv_out))
        # print(X_proj)
        X_gate = torch.sigmoid(self.gate_linear(X_conv_out))
        # print(X_gate)
        X_highway = torch.mul(X_gate, X_proj) + torch.mul((1 - X_gate), X_conv_out)
        # print(X_highway)
        return X_highway

if __name__ == "__main__":
    h = Highway(word_embedding_size=3)
    h.forward(torch.FloatTensor([[[1, 1, 1],[0, 0, 0]]]))
### END YOUR CODE


