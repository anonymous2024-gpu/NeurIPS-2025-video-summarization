import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        pe = self.pe[:x.size(1)].unsqueeze(0)
        return x + pe

class VideoEncoder(nn.Module):
    def __init__(self, embed_dim=1024, num_layers=6, num_heads=8, ff_dim=2048):
        super(VideoEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.pos_encoding = PositionalEncoding(embed_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=ff_dim, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # Ensure x has the right shape [batch_size, seq_len, embed_dim]
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer encoder
        return self.transformer_encoder(x)

class AudioEncoder(nn.Module):
    def __init__(self, embed_dim=1024, num_layers=3, num_heads=8, ff_dim=2048):
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(f"embed_dim and num_heads must be greater than 0. Got embed_dim={embed_dim} and num_heads={num_heads}")
        if embed_dim % num_heads != 0:
            embed_dim = (embed_dim // num_heads) * num_heads
            print(f"Adjusted embed_dim to {embed_dim} to be divisible by num_heads")
            
        self.embed_dim = embed_dim
        self.pos_encoding = PositionalEncoding(embed_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0) 
        
        x = self.pos_encoding(x)        
        return self.transformer_encoder(x)

class TextEncoder(nn.Module):
    """Text encoder using transformer architecture"""
    def __init__(self, embed_dim=1024, num_layers=3, num_heads=8, ff_dim=2048):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_encoding = PositionalEncoding(embed_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # Ensure x has the right shape [batch_size, seq_len, embed_dim]
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer encoder
        return self.transformer_encoder(x)

class CrossModalAttention(nn.Module):
    def __init__(self, d_model=1024, nhead=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, text=None, audio=None):
        device = next(self.parameters()).device
        query = query.to(device)

        # Ensure query is 3D
        if query.dim() == 2:
            query = query.unsqueeze(1)  # Shape: [batch, 1, d_model]
        elif query.dim() == 3 and query.shape[1] == 0:
            query = query.unsqueeze(1)

        # Determine reference sequence length from query
        ref_seq_len = query.shape[1]

        # Process text and audio for key-value
        combined_kv = []
        if text is not None:
            text = text.to(device)
            if text.dim() == 2:
                text = text.unsqueeze(1)
            # Align sequence length
            if text.shape[1] != ref_seq_len:
                text = text[:, :ref_seq_len, :] if text.shape[1] > ref_seq_len else \
                    text.repeat(1, ref_seq_len // text.shape[1] + 1, 1)[:, :ref_seq_len, :]
            combined_kv.append(text)

        if audio is not None:
            audio = audio.to(device)
            if audio.dim() == 2:
                audio = audio.unsqueeze(1)
            # Align sequence length
            if audio.shape[1] != ref_seq_len:
                audio = audio[:, :ref_seq_len, :] if audio.shape[1] > ref_seq_len else \
                    audio.repeat(1, ref_seq_len // audio.shape[1] + 1, 1)[:, :ref_seq_len, :]
            combined_kv.append(audio)

        # If no text or audio, use query as key-value
        if not combined_kv:
            combined_kv = query
        else:
            combined_kv = torch.cat(combined_kv, dim=1)

        # Ensure combined_kv has matching feature dimension
        if combined_kv.shape[-1] != query.shape[-1]:
            projection = nn.Linear(combined_kv.shape[-1], query.shape[-1]).to(device)
            combined_kv = projection(combined_kv)

        if combined_kv.dim() == 2:
            combined_kv = combined_kv.unsqueeze(1)

        # Apply multi-head attention
        attn_output, _ = self.multihead_attn(query, combined_kv, combined_kv)

        # Residual connection and normalization
        attn_output = self.layer_norm(attn_output + query)
        ffn_output = self.dropout(self.ffn(attn_output))
        output = self.layer_norm(ffn_output + attn_output)

        return output

class SummaryDecoder(nn.Module):
    def __init__(self, d_model=1024, nhead=8, num_layers=6, dropout=0.1, max_seq_len=500):
        super().__init__()
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=2048, 
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, context, target_sequence, tgt_mask=None):
        target_sequence = self.pos_encoding(target_sequence)
        output = self.decoder(target_sequence, context, tgt_mask=tgt_mask)
        output = self.norm(output)
        return self.fc_out(output)

class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)