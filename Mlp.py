from torch import nn
import torch
import numpy as np

"""Create multi layer perceptron model"""
class Mlp(nn.Module):
    def __init__(self, num_classes = 9):
        super().__init__()
        '''
        self.mlp = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(), 
            nn.Linear(512, num_classes))
        '''
        """
        To use with the Doc2Vec features extractor
        self.mlp = nn.Sequential(
            nn.Linear(300, 256),
            nn.ReLU(), 
            nn.Linear(256, num_classes))
        """
        self.norm1d = nn.BatchNorm1d(num_features = 2048, momentum = 0.99)
        self.lstm = nn.LSTM(input_size = 2048, hidden_size = 256, num_layers = 3, dropout=.5, batch_first = True)
        self.linear = nn.Linear(256, num_classes)
    
    def forward(self, x):
        #x = self.mlp(x)
        x = self.norm1d(x)
        x = x[:, None, :]
        #print(x.shape)
        ht, (hn, cn) = self.lstm(x)
        hn = hn.mean(dim = 0)
        y = self.linear(hn)
        return y