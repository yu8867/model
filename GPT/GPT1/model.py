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
    def __init__(self, hidden_size, drop_rate=0.1):
        super().__init__()
        self.dropout = nn.Dropout(drop_rate)
        self.hidden_size = hidden_size

    def forward(self, q, k, v, mask):
        attention = torch.matmul(q, k.transpose(2, 3)) / self.hidden_size**0.5 # 2:token 3:hidden
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-inf")) # 0 -> -inf

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
        self.attention = SDPA(hidden_size, drop_rate)
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(hidden_size, hidden_size)


    def forward(self, q, k, v, mask=None):
        N = q.size(0) # batch
        T = q.size(1) # token_size
        H = self.n_head # head数
        D = self.hidden_size // self.n_head

        q = self.fc_q(q).view(N, H, T, D)
        k = self.fc_k(k).view(N, H, T, D)
        v = self.fc_v(v).view(N, H, T, D)

        attn = self.attention(q, k, v, mask)
        x = attn.view(N, T, -1)
        x = self.fc(x)
        x = self.dropout(x)
        return x
    
class FeedForward(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(hidden_size, hidden_size*4)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size*4, hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, n_head, dropout_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.attn = MultiHeadAttention(n_head, hidden_size, dropout_rate)
        self.ff = FeedForward(hidden_size, dropout_rate)

        nn.init.normal_(self.norm1.weight, mean=0, std=0.02)
        nn.init.normal_(self.norm2.weight, mean=0, std=0.02)

    def forward(self, x):
        rx1 = x
        x = self.attn(x, x, x)
        x = self.norm1(x + rx1)

        rx2 = x
        x = self.ff(x)
        x = self.norm2(x + rx2)
        return x
    
class GPT1(nn.Module):
    def __init__(self, vocab_size, token, hidden_size, n_block, n_head, dropout_rate):
        super().__init__()
        self.vocab_size = vocab_size
        self.token = token
        self.hidden_size = hidden_size
        self.n_block = n_block
        self.n_head = n_head
        
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.positonal_encoding = PositionalEncoding(token, hidden_size)

        self.dropout = nn.Dropout(dropout_rate)
        self.block = nn.ModuleList(
            [TransformerBlock(hidden_size, n_head, dropout_rate) for _ in range(n_block)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size * token, vocab_size)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)

    def forward(self, x):
        x = self.token_embedding(x) + self.positonal_encoding(x)
        x = self.dropout(x)

        for block in self.block:
            x = block(x)

        x = self.norm(x)
        x = x.view(-1, self.hidden_size * self.token)
        x = self.fc(x)
        return x
        



if __name__=='__main__':
    batch_size = 1
    token = 10
    n_head = 8
    hidden_size = 128
    vocab_size = 10
    n_block = 2

    x = torch.randint(vocab_size, size=(batch_size, token))

    model = GPT1(vocab_size, token, hidden_size, n_block, n_head, dropout_rate=0.1)
    y = model(x)
    print(y)

    # print(y.shape)



    # batch_size = 1
    # token = 4
    # hidden_size = 5

    # torch.random.manual_seed(0)

    # pe = PositionalEncoding(token, hidden_size)
    # x = torch.randint(token, (batch_size, token))
    # print(x) # (batch_size, hidden_size)


    # # Positional encodingの実行
    # pe_encoding = pe(x)
    # # print(pe_encoding.size())

    # # Embdeeing
    # emb = nn.Embedding(token, hidden_size)
    # emb_encoding = emb(x)

    # y = pe_encoding + emb_encoding
    # print(y)

    # device = torch.device("mps")
    # print(device)

