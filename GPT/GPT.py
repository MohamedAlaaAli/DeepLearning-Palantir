import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# Hyperparameters
BATCH_SIZE = 32  # how many sequences to process at a single pass
BLOCK_SIZE = 8  # max context length to look into for making the predictions
max_iter = 5000
eval_interval = 500
learning_rate = 0.0001
device = "cuda" if torch.cuda.is_available() else "cpu"
n_embed = 32
dropout = 0.2
n_layers = 6

def load_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def vocab_info(text):
    chars = sorted(list(set(text)), reverse=False)
    vocab_size = len(chars)
    return chars, vocab_size

def split(data):
    n = int(0.9 * len(data))
    train_set = data[:n]
    test_set = data[n:]
    return train_set, test_set

def get_batch(split_type):
    data = train_set if split_type == "train" else test_set
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x, y

class BiGramLanguageModel(nn.Module):
    def __init__(self):
        super(BiGramLanguageModel, self).__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.positional_embedding_table = nn.Embedding(BLOCK_SIZE, n_embed)
        self.blocks = nn.Sequential(
            *[Block(n_embed, 4) for _ in range(n_layers)]
        )
        self.ln_head = nn.Linear(n_embed, vocab_size)

    def calc_loss(self, y_hat, y_true):
        return F.cross_entropy(y_hat, y_true)

    def forward(self, xb, yb=None):
        B, T = xb.shape
        token_embeddings = self.token_embedding_table(xb)
        pos_emb = self.positional_embedding_table(torch.arange(T, device=device))
        x = token_embeddings + pos_emb
        x = self.blocks(x)
        logits = self.ln_head(x)

        if yb is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            yb = yb.view(B * T)
            loss = self.calc_loss(logits, yb)
            return logits, loss
        
        return logits, None

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class Head(nn.Module):
    def __init__(self, head_size):
        super(Head, self).__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones((BLOCK_SIZE, BLOCK_SIZE))))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        w = q @ k.transpose(-2, -1) * C**-0.5
        w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        v = self.value(x)
        return w @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.projection = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.projection(out))

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super(FeedForward, self).__init__()
        self.ffnn = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.ffnn(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super(Block, self).__init__()
        head_size = n_embed // n_head
        self.attention = MultiHeadAttention(n_head, head_size)
        self.ffnn = FeedForward(n_embed)
        self.norm1 = nn.LayerNorm(n_embed)
        self.norm2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffnn(self.norm2(x))
        return x

text = load_data('input.txt')
chars, vocab_size = vocab_info(text)

str_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_str = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [str_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_str[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
train_set, test_set = split(data)

m = BiGramLanguageModel().to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for epoch in range(max_iter + 1):
    if epoch % eval_interval == 0:
        m.eval()
        with torch.no_grad():
            xb, yb = get_batch("test")
            logits, loss = m(xb.to(device), yb.to(device))
            print(f"Epoch: {epoch}, Validation Loss: {loss.item()} at device {device}")
    else:
        m.train()
        xb, yb = get_batch("train")
        logits, loss = m(xb.to(device), yb.to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(f"Epoch: {epoch}, Training Loss: {loss.item()} at device {device}")

print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=10000)[0].tolist()))
