
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PositionwiseFeedForward(nn.Module):
    # Part 3.3 of the paper
    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff )
        self.act =  nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EmbeddingLayer(nn.Module):
    # Part 3.4 of the paper
    def __init__(self, vocab_size, d_model):
        super(EmbeddingLayer, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * np.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    # Part 3.5 of the paper
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class LayerAddNorm(nn.Module):
    # Part 5.4
    def __init__(self, d_model, dropout):
        super(LayerAddNorm, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, residual_connection):
        return self.dropout(self.norm(x + residual_connection))

class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, d_model, dropout):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % heads == 0
        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads # 512/8 = 64

        self.dropout = nn.Dropout(dropout)
        self.Q_proj = nn.Linear(d_model, d_model, bias=False)
        self.K_proj = nn.Linear(d_model, d_model, bias=False)
        self.V_proj = nn.Linear(d_model, d_model, bias=False)
        self.final_linear = nn.Linear(d_model, d_model)

        self.attention_weights = None

    def scaled_dot_product_attention(self, query, key, value, mask):
        # Figure 2 (left)
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            # Same mask applied to all heads.
            scores = scores.masked_fill(mask == 0, np.NINF) # Negative infinity

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context_vector = torch.matmul(attention_weights, value)
        return context_vector, attention_weights

    def forward(self, query, key, value, mask) -> None:
        if mask is not None:
            mask = mask.unsqueeze(1)

        # Get number of training examples
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        query = self.Q_proj(query)  # (N, query_len, d_model)
        key = self.K_proj(key)      # (N, key_len, d_model)
        value = self.V_proj(value)  # (N, value_len, d_model)

        # Split the embedding into self.heads pieces
        query = query.view(N, query_len, self.heads, self.d_k).transpose(1, 2) # (N, heads, query_len, d_k)
        key = key.view(N, key_len, self.heads, self.d_k).transpose(1, 2)     # (N, heads, key_len, d_k)
        value = value.view(N, value_len, self.heads, self.d_k).transpose(1, 2) # (N, heads, value_len, d_k)

        # Compute the attention weights
        context_vector, self.attention_weights = self.scaled_dot_product_attention(query, key, value, mask)

        # Concatenate the different heads
        context_vector = context_vector.transpose(1, 2).reshape(N, query_len, self.d_model)
        # Apply final linear layer
        context_vector = self.final_linear(context_vector)

        del query, key, value
        return context_vector

class TransformerBlock(nn.Module):

    def __init__(self, heads, d_model, d_ff, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(heads, d_model, dropout)
        self.add_norm1 = LayerAddNorm(d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.add_norm2 = LayerAddNorm(d_model, dropout)

    def forward(self, residual, query, key, value, mask) -> None:
        x = self.attention(query, key, value, mask)
        x = self.add_norm1(x, residual)
        out = self.feed_forward(x)
        out = self.add_norm2(out, x)
        return out

class Encoder(nn.Module):
    def __init__(self, N, layer):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)

    def forward(self, inp_embed, mask):
        for layer in self.layers:
            inp_embed = layer(residual=inp_embed, query=inp_embed, key=inp_embed, value=inp_embed, mask=mask)
        return inp_embed

class Decoder(nn.Module):
    def __init__(self, N, layer):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)

    def forward(self, out_embed, encoder_output, src_mask, tgt_mask):

        for layer in self.layers:
            # Masked Multi-Headed Attention
            out = layer[0](query=out_embed, key=out_embed, value=out_embed, mask=tgt_mask)
            # Layer Add Norm
            out = layer[1](out, out_embed)
            # Transformer Block
            out_embed = layer[2](residual=out, query=out, key=encoder_output, value=encoder_output, mask=src_mask)
        return out_embed

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, N=6, heads=8, d_model=512, d_ff=2048, dropout=0.1):
        super(Transformer, self).__init__()

        self.d_model = d_model

        # FIXME: Using EmbeddingLayer doesn't work anymore. Some pist to explore: https://datascience.stackexchange.com/a/87909/87155
        self.input_embedding = nn.Sequential(
            nn.Embedding(src_vocab_size, d_model),
            # EmbeddingLayer(src_vocab_size, d_model),
            PositionalEncoding(d_model, dropout)
        )
        self.output_embedding = nn.Sequential(
            nn.Embedding(tgt_vocab_size, d_model),
            # EmbeddingLayer(tgt_vocab_size, d_model),
            PositionalEncoding(d_model, dropout)
        )
        self.encoder = Encoder(N, TransformerBlock(heads, d_model, d_ff, dropout))
        self.decoder = Decoder(
            N, 
            nn.Sequential(
                MultiHeadedAttention(heads, d_model, dropout),
                LayerAddNorm(d_model, dropout),
                TransformerBlock(heads, d_model, d_ff, dropout)
            )
        )

        self.predict = nn.Sequential(
            nn.Linear(d_model, tgt_vocab_size),
            nn.LogSoftmax(dim=-1)
        )
    
    def encode(self, src, src_mask):
        return self.encoder(self.input_embedding(src), src_mask)
    
    def decode(self, encoder_output, tgt, src_mask, tgt_mask):
        return self.decoder(self.output_embedding(tgt), encoder_output, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), tgt, src_mask, tgt_mask)