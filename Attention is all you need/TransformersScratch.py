import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, heads):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads

        assert (self.heads*self.head_dim == self.embed_dim),"Make sure that the embedding dimension needs to be divisible by the number of heads"

        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.query = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_dim)

    
    def forward(self, values, query, key, mask):
        N = query.shape[0]
        val_len, key_len, query_len = values.shape[1], key.shape[1], query.shape[1]

        values = values.reshape(N, val_len, self.heads, self.head_dim)
        key = key.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        key = self.keys(key)
        query = self.query(query)

        sim = torch.einsum("nqhd,nkhd->nhqk", [query, key])

        if mask is not None:
            sim = sim.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(sim/torch.sqrt(self.embed_dim), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim)

        return self.fc_out(out)
    

class TransformerBlock(nn.module):
    def __init__(self, embed_dim, heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_dim, heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffnn= nn.Sequential(
            nn.Linear(embed_dim, ff_dim*embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim*embed_dim, embed_dim),
        )

        self.dropout = nn.Dropout(dropout_rate)
      

    def forward(self, values, query, key, mask):
        attention = self.attention(values, key, query, mask)
        x = self.dropout(self.norm1(attention+query))
        fwd = self.ffnn(x)
        
        return self.dropout(self.norm2(fwd+x))


class Encoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            embed_size,
            n_layers,
            heads, 
            device, 
            ff_dim, 
            dropout_rate=0.1,
            max_length=100
        ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_size,
                    heads=heads,
                    ff_dim=ff_dim,
                    dropout_rate=dropout_rate
                )
                for _ in range(n_layers)  # n_layers is the number of transformer blocks in the encoder.
            ]
        )

        self.dropout = nn.Dropout(dropout_rate)
    

    def forward(self, x, mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).expand(N, seq_len).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding)
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, ff_dim, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer=TransformerBlock(embed_size, heads, ff_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self ,x, value, key, src_mask, trg_mask):
        attention = self.attention(x,x,x,trg_mask )
        query = self.dropout(self.norm(attention+x))
        out = self.transformer(value, query, key, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
            self,
            trg_vocab_size,
            embed_size,
            n_layers,
            heads, 
            device, 
            ff_dim, 
            dropout_rate=0.1,
            max_length=100
            ):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embed_size=embed_size,
                    heads=heads,
                    ff_dim=ff_dim,
                    dropout=dropout_rate,
                    device=device
                ) for _ in range(n_layers)
            ]
        )   

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_out, src_mask, target_mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, enc_out, enc_out, src_mask, target_mask)
        return self.fc_out(out)


class Transformer(nn.Module):

    def __init__(
            self,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            target_pad_idx,
            embed_size=256,
            ff_dim = 4,
            dropout=0,
            device="cuda",
            max_len=100
            ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size=src_vocab_size,
            embed_size=embed_size,
            n_layers=6,
            heads=8,
            device=device,
            ff_dim=ff_dim,
            dropout_rate=dropout,
            max_length=max_len
            )

        self.decoder = Decoder(
            trg_vocab_size=trg_vocab_size,
            embed_size=embed_size,
            n_layers=6,
            heads=8,
            device=device,
            ff_dim=ff_dim,
            dropout_rate=dropout,
            max_length=max_len
            )
        
        self.src_pad_idx = src_pad_idx
        self.target_pad_idx = target_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src!= self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)
    
    def make_tgt_mask(self, tgt):
        tgt_mask = (tgt!= self.target_pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.shape[1]
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 0, float('-inf')).tril(-tgt_len).to(self.device)
        return tgt_mask.to(self.device)

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        enc_out = self.encoder(src, src_mask)
        out = self.decoder(tgt, enc_out, src_mask, tgt_mask)

        return out
    


            









    