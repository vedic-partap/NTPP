import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NTPP(nn.Module):
    
    def __init__(self, args, output_layer_size):
        super(NTPP,self).__init__()
        self.hidden_size = args['h']
        self.num_layers = args['nl']
        self.rnn_step = nn.RNN(args['element_size'], args['h'], args['nl'], batch_first=True)
        self.fc = nn.Linear(args['h'], output_layer_size) # output_layer_size = number of host

    def forward(self,x):

        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out