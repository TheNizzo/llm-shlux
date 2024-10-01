import torch
import torch.nn as nn
import sys
sys.path.append('../')
from attention_mechanism.multi_head_attention import MultiHeadAttention

class LayerNorm(nn.Module):
    """
    Layer normalization module

    args:
    - emb_dim: the dimension of the embedding
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    """
    GELU activation function
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    """
    Feed-forward layer
    """
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg["n_embd"], 4 * cfg["n_embd"]),
            GELU(),
            nn.Linear(4 * cfg["n_embd"], cfg["n_embd"]))

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    Transformer block
    """
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["n_embd"],
            d_out=cfg["n_embd"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["dropout"], bias=cfg["bias"])
        self.ln1 = LayerNorm(cfg["n_embd"])
        self.ln2 = LayerNorm(cfg["n_embd"])
        self.ff = FeedForward(cfg)
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.ln1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.dropout(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.ln2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut  # Add the original input back
        return x


class GPT(nn.Module):
    """
    GPT model
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["n_embd"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["n_embd"])
        self.dropout = nn.Dropout(cfg["dropout"])
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.ln_f = LayerNorm(cfg["n_embd"])
        self.head = nn.Linear(cfg["n_embd"], cfg["vocab_size"], bias=False)
    
    def forward(self, x):
        _, seq_len = x.shape
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=x.device))
        x = tok_emb + pos_emb
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.ln_f(x)
        return self.head(x)


