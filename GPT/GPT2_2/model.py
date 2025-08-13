import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, token, hidden_size):
        super().__init__()

        pe = torch.zeros(token, hidden_size)

        for pos in range(token):
            for i in range(hidden_size):
                if i % 2 == 0:
                    pe[pos, i] = math.sin(pos/(10000**((2*i)/hidden_size)))
                else:
                    pe[pos, i] = math.cos(pos/(10000**((2*(i-1))/hidden_size)))

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    

class SDPA(nn.Module):
    def __init__(self, hidden_size, n_head, drop_rate=0.1):
        super().__init__()
        self.dropout = nn.Dropout(drop_rate)
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // n_head

    def forward(self, q, k, v, mask):
        attention = torch.matmul(q, k.transpose(2, 3)) / (self.head_dim**0.5) # 2:token 3:hidden
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9) # 0 -> -inf

        # print(attention)

        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)
        attention = torch.matmul(attention, v)

        return attention
    

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, hidden_size, drop_rate=0.1):
        super().__init__()
        self.n_head = n_head
        self.hidden_size = hidden_size
        self.fc_q = nn.Linear(hidden_size, hidden_size)
        self.fc_k = nn.Linear(hidden_size, hidden_size)
        self.fc_v = nn.Linear(hidden_size, hidden_size)
        self.attention = SDPA(hidden_size, n_head, drop_rate)
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(hidden_size, hidden_size)


    def forward(self, q, k, v, past, mask=None):
        N = q.size(0) # batch
        T = q.size(1) # token_size
        H = self.n_head # head数
        D = self.hidden_size // self.n_head

        q = self.fc_q(q).view(N, T, H, D).transpose(1, 2)
        k = self.fc_k(k).view(N, T, H, D).transpose(1, 2)
        v = self.fc_v(v).view(N, T, H, D).transpose(1, 2)

        if past is not None:
            past_k, past_v = past
            k = torch.cat([k, past_k], dim=-2)
            v = torch.cat([v, past_v], dim=-2)

        attn = self.attention(q, k, v, mask)
        x = attn.view(N, T, -1)
        x = self.fc(x)
        x = self.dropout(x)

        past = torch.stack([k.detach(), v.detach()])

        return x, past
    
# class FeedForward(nn.Module):
#     def __init__(self, hidden_size, dropout_rate=0.1):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.fc1 = nn.Linear(hidden_size, hidden_size*4)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.fc2 = nn.Linear(hidden_size*4, hidden_size)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.gelu(x)
#         x = self.fc2(x)
#         x = self.dropout(x)
#         return x

class SwiGLU(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.1, mult=4):
        super().__init__()
        inner = int(hidden_size * mult)
        self.w1 = nn.Linear(hidden_size, inner, bias=False)
        self.w2 = nn.Linear(hidden_size, inner, bias=False)
        self.w3 = nn.Linear(inner, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        a = F.silu(self.w1(x))
        b = self.w2(x)
        x = self.w3(a*b)
        x = self.dropout(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, device, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(hidden_size, device=device))

    def forward(self, x):
        x = x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x = x * self.scale
        return x 
    
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, n_head, device, dropout_rate=0.1):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, device)
        self.norm2 = RMSNorm(hidden_size, device)
        self.attn = MultiHeadAttention(n_head, hidden_size, dropout_rate)
        self.ff = SwiGLU(hidden_size, dropout_rate)

    def forward(self, x, past, mask):
        rx1 = x
        x = self.norm1(x)
        x, past = self.attn(x, x, x, past, mask)
        x = x + rx1

        rx2 = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + rx2
        return x, past
    
class GPT2_2(nn.Module):
    def __init__(self, vocab_size, token, hidden_size, n_block, n_head, device, dropout_rate):
        super().__init__()
        self.vocab_size = vocab_size
        self.token = token
        self.hidden_size = hidden_size
        self.n_block = n_block
        self.n_head = n_head
        self.device = device
        
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.positonal_encoding = PositionalEncoding(token, hidden_size)

        self.dropout = nn.Dropout(dropout_rate)
        self.block = nn.ModuleList(
            [TransformerBlock(hidden_size, n_head, device, dropout_rate) for _ in range(n_block)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size * token, vocab_size)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)

    def _causal_mask(self, token):
        causal_mask = torch.ones((token, token), device=self.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask == 0
        return causal_mask * 1

    def forward(self, x, kv_cache=None):
        x = self.token_embedding(x) + self.positonal_encoding(x)
        x = self.dropout(x)
        mask = self._causal_mask(self.token)

        if kv_cache is not None:
            past = torch.unbind(kv_cache, dim=1)
        else:
            past = [None] * self.n_block

        presents = []
        for block, past_block in zip(self.block, past):
            x, present = block(x, past_block, mask)
            presents.append(present)

        x = self.norm(x)
        x = x.view(-1, self.hidden_size * self.token)
        x = self.fc(x)
        return x, torch.stack(presents, dim=1)