import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        embed_size = 2304
        super().__init__(vocab_size, embed_size, padding_idx=0)
