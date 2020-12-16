#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.char_embedding_size = char_embedding_size
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(input_size=self.char_embedding_size, hidden_size=self.hidden_size)
        self.char_output_projection = nn.Linear(in_features=hidden_size, out_features=len(self.target_vocab.char2id), bias=True)
        self.decoderCharEmb = nn.Embedding(num_embeddings=len(self.target_vocab.char2id), embedding_dim=self.char_embedding_size, padding_idx=self.target_vocab.char2id['<pad>'])
        self.loss = nn.CrossEntropyLoss(ignore_index=self.target_vocab.char2id['<pad>'], reduction="sum")
        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        input_embedded = self.decoderCharEmb(input) # out of size (len, batch, char_embed_size)
        if dec_hidden is not None:
            h_tm1, c_tm1 = dec_hidden
            output, (h_n, c_n) = self.charDecoder(input_embedded, (h_tm1, c_tm1))  # h_t is 1 X batch X hidden_size, output is seq_len X batch X hidden_size
        else:
            output, (h_n, c_n) = self.charDecoder(input=input_embedded)

        output.permute(1, 0, 2)
        s = self.char_output_projection(output) # out of size batch X seq_len X vocab_char_size
        s.permute(1, 0, 2)
        return s, (h_n, c_n)
        ### END YOUR CODE


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch, for every character in the sequence.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        char_sequence_input_for_train = char_sequence[:-1,] # of size (len-1) X batch
        char_sequence_for_target_evaluation = char_sequence[1:,] # of size (len-1) X batch
        s, (h_n, c_n) = self.forward(char_sequence_input_for_train, dec_hidden) # s is len-1 X batch X vocab_char_size
        vocab_char_size = s.size(2)
        s = s.contiguous().view(-1, vocab_char_size) # s is now (num_output chars across entire sentences batch X vocab_char_size)
        cr_loss = self.loss(s, char_sequence_for_target_evaluation.contiguous().view(-1))
        return cr_loss
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        batch_size = initialStates[0].size(1)
        current_lstm_state = initialStates
        output_words_batch = [] # list of max_length X batch_size chars
        current_character_idx = torch.tensor([self.target_vocab.start_of_word]*batch_size, device=device) # of size batch

        for t in range(max_length):
            #input of size len_seq=1 X batch X char_embed_size
            current_character_embed = self.decoderCharEmb(current_character_idx) # size is batch X char_embed_size
            current_character_embed = current_character_embed.unsqueeze(0) # size is 1 X batch X char_embed_size
            output, (h_n, c_n) = self.charDecoder(current_character_embed, current_lstm_state)# output, h_n, c_n is 1 X batch X hidden_size
            current_lstm_state = (h_n, c_n)
            s = self.char_output_projection(output.squeeze(0)) # This will be a batch X vocab_size
            p = torch.nn.functional.softmax(s, dim=1) # This will be batch x vocab_size (probs)
            _, current_character_idx = p.max(1) # Output of size batch
            current_output_words = [self.target_vocab.id2char[id] for id in current_character_idx.tolist()]
            output_words_batch.append(current_output_words)

        output_words_batch = list(map(list, zip(*output_words_batch))) #makes ir batch X max_length (of chars)

        end_token = self.target_vocab.id2char[self.target_vocab.end_of_word]
        for idx, output_words_sent in enumerate(output_words_batch):
            try:
                index_value = output_words_sent.index(end_token)
                output_words_batch[idx] = output_words_sent[:index_value]
            except ValueError:
                None
        output_words_batch = [''.join(output_words_sent) for output_words_sent in output_words_batch]
        return output_words_batch
        ### END YOUR CODE

