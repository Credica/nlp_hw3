"""
RNN-based Neural Machine Translation Model
Based on PyTorch Seq2Seq Tutorial 3: Attention Mechanism

This implements:
- Bidirectional GRU Encoder
- Bahdanau Attention (Additive Attention)
- GRU Decoder with attention
- Teacher Forcing during training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import random

class Encoder(nn.Module):
    """Bidirectional GRU Encoder"""

    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        # Combine bidirectional hidden states
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src):
        # src shape: [src_len, batch_size]

        # Embedding
        embedded = self.dropout(self.embedding(src))
        # embedded shape: [src_len, batch_size, emb_dim]

        # RNN
        outputs, hidden = self.rnn(embedded)
        # outputs shape: [src_len, batch_size, hidden_dim * 2]
        # hidden shape: [n_layers * 2, batch_size, hidden_dim]

        # Combine bidirectional hidden states
        # hidden shape: [n_layers, batch_size, hidden_dim * 2] -> [n_layers, batch_size, hidden_dim]
        hidden = hidden.view(self.n_layers, 2, -1, self.hidden_dim)
        hidden = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)

        # Initial decoder hidden state
        hidden = torch.tanh(self.fc_hidden(hidden))
        cell = torch.tanh(self.fc_cell(hidden))

        return outputs, hidden, cell


class Attention(nn.Module):
    """Bahdanau Attention (Additive Attention)"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear((hidden_dim * 2) + hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden shape: [batch_size, hidden_dim]
        # encoder_outputs shape: [src_len, batch_size, hidden_dim * 2]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # Repeat hidden state for each source position
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # hidden shape: [batch_size, src_len, hidden_dim]

        # Permute encoder outputs to [batch_size, src_len, hidden_dim * 2]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # Concatenate hidden state with encoder outputs
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy shape: [batch_size, src_len, hidden_dim]

        # Compute attention scores
        attention = self.v(energy).squeeze(2)
        # attention shape: [batch_size, src_len]

        # Softmax to get attention weights
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    """GRU Decoder with Attention"""

    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((hidden_dim * 2) + emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear((hidden_dim * 2) + hidden_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input shape: [batch_size]
        # hidden shape: [n_layers, batch_size, hidden_dim]
        # encoder_outputs shape: [src_len, batch_size, hidden_dim * 2]

        input = input.unsqueeze(0)
        # input shape: [1, batch_size]

        # Embedding
        embedded = self.dropout(self.embedding(input))
        # embedded shape: [1, batch_size, emb_dim]

        # Compute attention weights
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        # attn_weights shape: [batch_size, src_len]

        # Reshape for batch matrix multiplication
        attn_weights = attn_weights.unsqueeze(1)
        # attn_weights shape: [batch_size, 1, src_len]

        # Apply attention to encoder outputs
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs shape: [batch_size, src_len, hidden_dim * 2]

        weighted = torch.bmm(attn_weights, encoder_outputs)
        # weighted shape: [batch_size, 1, hidden_dim * 2]

        weighted = weighted.permute(1, 0, 2)
        # weighted shape: [1, batch_size, hidden_dim * 2]

        # Concatenate embedding with weighted context
        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input shape: [1, batch_size, emb_dim + hidden_dim * 2]

        # RNN
        output, hidden = self.rnn(rnn_input, hidden)
        # output shape: [1, batch_size, hidden_dim]
        # hidden shape: [n_layers, batch_size, hidden_dim]

        # Prediction
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        # prediction shape: [batch_size, output_dim]

        return prediction, hidden, attn_weights.squeeze(1)


class Seq2Seq(nn.Module):
    """Seq2Seq Model with Encoder, Decoder, and Attention"""

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src shape: [src_len, batch_size]
        # trg shape: [trg_len, batch_size]

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # Initialize outputs tensor
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # Encoder
        encoder_outputs, hidden, cell = self.encoder(src)

        # First input to decoder is <sos> token
        input = trg[0, :]

        # Decoder
        for t in range(1, trg_len):
            # Decode one time step
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)

            # Store prediction
            outputs[t] = output

            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs


def build_model():
    """Build and return the complete model"""

    # Create attention mechanism
    attention = Attention(config.HIDDEN_DIM)

    # Create encoder
    encoder = Encoder(
        input_dim=config.INPUT_DIM,
        emb_dim=config.ENC_EMB_DIM,
        hidden_dim=config.HIDDEN_DIM,
        n_layers=config.N_LAYERS,
        dropout=config.ENC_DROPOUT
    )

    # Create decoder
    decoder = Decoder(
        output_dim=config.OUTPUT_DIM,
        emb_dim=config.DEC_EMB_DIM,
        hidden_dim=config.HIDDEN_DIM,
        n_layers=config.N_LAYERS,
        dropout=config.DEC_DROPOUT,
        attention=attention
    )

    # Create seq2seq model
    model = Seq2Seq(encoder, decoder, config.device).to(config.device)

    return model


def count_parameters(model):
    """Count number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    """Initialize model weights"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.1, 0.1)
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
