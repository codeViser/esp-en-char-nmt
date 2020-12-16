#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1e
import torch
import torch.nn as nn
import torch.nn.utils

class CNN(nn.Module):
    def __init__(self, num_input_channels, num_out_channels, input_embed_size, kernel_size=5, stride_value=1, padding=0, dilation=1, bias=True):
        super(CNN, self).__init__()
        self.num_input_channels = num_input_channels # has to be of size: embed_char_size
        self.num_out_channels = num_out_channels # has to be of size: embed_word_size
        self.input_embed_size = input_embed_size # will be embed_char_size
        self.kernal_size = kernel_size
        self.stride_value = stride_value
        self.padding = padding
        self.dilation = dilation
        self.conv_output_embed_size = int(((self.input_embed_size + 2* self.padding - self.dilation * (self.kernal_size -1) - 1) / self.stride_value) + 1)
        self.bias = bias
        self.conv_layer = nn.Conv1d(in_channels=self.num_input_channels, out_channels=self.num_out_channels, kernel_size=self.kernal_size, stride=self.stride_value, padding=self.padding, dilation= self.dilation, bias=self.bias)
        self.relu = nn.ReLU()
        self.max_pool_1d = nn.MaxPool1d(kernel_size=self.conv_output_embed_size, stride=self.stride_value, padding=self.padding, dilation=self.dilation)

    def forward(self, X_reshaped) -> torch.Tensor:
        # X_reshaped is batched here of dimension: batch X * (embed_char_size here) X m_word
        X_convolved = self.relu(self.conv_layer(X_reshaped)) # output would be of shape N X (num_out_channels) X (conv_output_embed_size)
        # print(X_convolved)
        X_conv_out = self.max_pool_1d(X_convolved) # output would be of shape N X (num_out_channels) X 1
        # print(X_conv_out)
        X_conv_out = torch.squeeze(X_conv_out, 2) # output would be of shape N X (num_out_channels)
        # print(X_conv_out)
        return X_conv_out

if __name__ == "__main__":
    c = CNN(num_input_channels=3, num_out_channels=2, input_embed_size=5)
    c.forward(torch.FloatTensor([[[1, 1, 1, 1, 1],[0, 0, 0, 0, 0], [2, 2, 2, 2, 2]]]))
### END YOUR CODE

