import jittor as jt
from jittor import nn
from unet import SiLU

class ConditionalEmbedding(nn.Module):
    def __init__(self, num_labels:int, d_model:int, dim:int):
        assert d_model % 2 == 0
        super().__init__()
        self.condEmbedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_labels + 1, embedding_dim=d_model, padding_idx=0),
            nn.Linear(d_model, dim),
            SiLU(),
            nn.Linear(dim, dim),
        )

    def execute(self, t):
        emb = self.condEmbedding(t)
        return emb
