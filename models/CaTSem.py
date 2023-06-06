import math
import torch
import timm

class ViTDecapitated(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.patch_embed = vit.patch_embed
        self.cls_token = vit.cls_token
        self.pos_drop = vit.pos_drop
        self.pos_embed = vit.pos_embed
        self.blocks = torch.nn.Sequential(*list(vit.blocks))
        self.norm = vit.norm
    def forward(self, images):
        x = self.patch_embed(images)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 1:]

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model=768, dropout=0.1, max_len=10):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransDecoder(torch.nn.Module):
    def __init__(self, vocab_size, semantic_size, nhead, forward_size, num_layers, max_len):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 768)
        self.we = torch.nn.Linear(768, 768)
        self.ws = torch.nn.Linear(semantic_size, 768)
        decoder_layer = torch.nn.TransformerDecoderLayer(768, nhead, forward_size, activation='gelu')
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers)
        self.linear = torch.nn.Linear(768, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=2)
        self.pos_enc = PositionalEncoding(max_len=max_len)
    def forward(self, tgt, memory, semantics, tgt_mask, tgt_key_padding_mask):
        memory = memory.permute(1,0,2)
        embed = self.embedding(tgt)
        x1 = self.we(embed)
        x2 = self.ws(semantics)
        x = torch.mul(x1, x2.unsqueeze(1))
        tgt = x.permute(1,0,2)
        tgt = self.pos_enc(tgt * math.sqrt(768))
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        output = output.permute(1,0,2)
        output = self.linear(output)
        return self.softmax(output)

class CaT(torch.nn.Module):
    def __init__(self, vocab_size, semantic_size, nhead, forward_size, num_layers, max_len):
        super().__init__()
        self.encoder = ViTDecapitated()
        self.decoder = TransDecoder(vocab_size, semantic_size, nhead, forward_size, num_layers, max_len)
    def forward(self, images, tgt, semantics, tgt_mask, tgt_key_padding_mask):
        memory = self.encoder(images)
        output = self.decoder(tgt, memory, semantics, tgt_mask, tgt_key_padding_mask)
        return output
    def decode(self, images, tgt, semantics, tgt_mask, memory=None):
        if memory == None:
            memory = self.encoder(images)
        output = self.decoder(tgt, memory, semantics, tgt_mask, None)
        return memory, output.squeeze(0)

