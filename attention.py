import math
import torch 
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x) :
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class TransformerLayer(nn.Module):
    def __init__(self,nlayers, embed_size, num_heads, ff_hidden_size, dropout=0.1, max_len=5000):
        super(TransformerLayer, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_size, dropout, max_len)
        norm_layer = nn.LayerNorm(embed_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=ff_hidden_size, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers, norm=norm_layer)

    def forward(self, x):
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)

        return x