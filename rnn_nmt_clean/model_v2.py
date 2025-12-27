import torch
import torch.nn as nn
import torch.nn.functional as F
import config_v2 as config
import random

class Encoder(nn.Module):
    """Unidirectional GRU Encoder"""

    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, dropout=dropout, bidirectional=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src shape: [src_len, batch_size]

        # Embedding
        embedded = self.dropout(self.embedding(src))
        # embedded shape: [src_len, batch_size, emb_dim]

        # RNN
        outputs, hidden = self.rnn(embedded)
        # outputs shape: [src_len, batch_size, hidden_dim]
        # hidden shape: [n_layers, batch_size, hidden_dim]

        return outputs, hidden


class Attention(nn.Module):
    """Attention Mechanism with multiple types"""

    def __init__(self, hidden_dim, attention_type='additive'):
        super().__init__()
        self.attention_type = attention_type
        self.hidden_dim = hidden_dim

        if attention_type == 'additive':
            # Bahdanau (Additive) Attention
            self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
            self.v = nn.Linear(hidden_dim, 1, bias=False)
        elif attention_type == 'multiplicative':
            # Luong (Multiplicative) Attention
            self.attn = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif attention_type == 'dot':
            # Dot Product Attention (scaled)
            pass  # No parameters needed
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

    def forward(self, hidden, encoder_outputs):
        # hidden shape: [batch_size, hidden_dim] (last layer hidden state)
        # encoder_outputs shape: [src_len, batch_size, hidden_dim]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        if self.attention_type == 'additive':
            # Bahdanau (Additive) Attention
            # Repeat hidden state for each source position
            hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
            # hidden shape: [batch_size, src_len, hidden_dim]

            # Permute encoder outputs to [batch_size, src_len, hidden_dim]
            encoder_outputs = encoder_outputs.permute(1, 0, 2)

            # Concatenate hidden state with encoder outputs
            energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
            # energy shape: [batch_size, src_len, hidden_dim]

            # Compute attention scores
            attention = self.v(energy).squeeze(2)
            # attention shape: [batch_size, src_len]

        elif self.attention_type == 'multiplicative':
            # Luong (Multiplicative) Attention
            hidden = hidden.unsqueeze(1)
            # hidden shape: [batch_size, 1, hidden_dim]

            # Transform encoder outputs
            encoder_outputs = encoder_outputs.permute(1, 0, 2)
            # encoder_outputs shape: [batch_size, src_len, hidden_dim]

            # Compute energy (hidden * W * encoder_outputs)
            energy = torch.bmm(hidden, encoder_outputs.transpose(1, 2)).squeeze(1)
            # energy shape: [batch_size, src_len]

            attention = energy

        elif self.attention_type == 'dot':
            # Dot Product Attention (scaled)
            hidden = hidden.unsqueeze(1)
            # hidden shape: [batch_size, 1, hidden_dim]

            encoder_outputs = encoder_outputs.permute(1, 0, 2)
            # encoder_outputs shape: [batch_size, src_len, hidden_dim]

            # Compute dot product (scaled)
            energy = torch.bmm(hidden, encoder_outputs.transpose(1, 2)).squeeze(1)
            # energy shape: [batch_size, src_len]

            # Scale by sqrt(hidden_dim)
            energy = energy / (self.hidden_dim ** 0.5)

            attention = energy

        # Softmax to get attention weights
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    """Unidirectional GRU Decoder (2 layers) with Attention"""

    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout, attention_type='additive'):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.attention = Attention(hidden_dim, attention_type)
        self.rnn = nn.GRU(hidden_dim + emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim + hidden_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input shape: [batch_size]
        # hidden shape: [n_layers, batch_size, hidden_dim]
        # encoder_outputs shape: [src_len, batch_size, hidden_dim]

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
        # encoder_outputs shape: [batch_size, src_len, hidden_dim]

        weighted = torch.bmm(attn_weights, encoder_outputs)
        # weighted shape: [batch_size, 1, hidden_dim]

        weighted = weighted.permute(1, 0, 2)
        # weighted shape: [1, batch_size, hidden_dim]

        # Concatenate embedding with weighted context
        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input shape: [1, batch_size, emb_dim + hidden_dim]

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
    """Seq2Seq Model with configurable components"""

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
        encoder_outputs, hidden = self.encoder(src)

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


def build_model(attention_type='additive', teacher_forcing_ratio=0.5):
    """Build and return the complete model"""

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
        attention_type=attention_type
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
