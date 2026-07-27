import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """ one self-attention head """

    def __init__(self, n_embed, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self attention in parallel """

    def __init__(self, n_embed, num_heads, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed, head_size, block_size, dropout) for _ in range(num_heads)])
        self.ffn = nn.Linear(num_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.ffn(out))
        return out


class FeedForward(nn.Module):
    """ simple feed forward network with non-linearity """

    def __init__(self, n_embed, dropout):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ffn(x)


class Block(nn.Module):
    """ transformer block: self-attention + feed-forward, each with a residual connection """

    def __init__(self, n_embed, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_embed, n_head, head_size, block_size, dropout)
        self.ffn = FeedForward(n_embed, dropout)
        self.ly1 = nn.LayerNorm(n_embed)
        self.ly2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ly1(x))
        x = x + self.ffn(self.ly2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size, n_embed, n_head, n_layer, dropout, use_pos_embed=True):
        super().__init__()
        self.block_size = block_size
        self.use_pos_embed = use_pos_embed

        self.token_embeds = nn.Embedding(vocab_size, n_embed)
        if use_pos_embed:
            self.pos_embeds = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embed)
        self.ffn = nn.Linear(n_embed, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        x = self.token_embeds(idx)
        if self.use_pos_embed:
            pos_emb = self.pos_embeds(torch.arange(T, device=idx.device))
            x = x + pos_emb

        x = self.blocks(x)
        x = self.ln(x)
        logits = self.ffn(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_context = idx[:, -self.block_size:]
            logits, _ = self(idx_context)
            logits = logits[:, -1]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
