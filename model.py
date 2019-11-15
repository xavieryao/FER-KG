from torch import nn
import torch
from torch.nn import functional as F
from kg_data import FB5KDataset


class RankingLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    @staticmethod
    def score(s, r, t):
        return F.normalize(s + r - t, 2, -1)

    def forward(self, batch):
        pos_score = self.score(batch["s"], batch["r"], batch["o"])
        neg_score = self.score(batch["s'"], batch["r"], batch["o'"])
        return self.margin + pos_score - neg_score


class TransEModel(nn.Module):
    def __init__(self, num_entities, num_relations, embed_dim):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embed_dim = embed_dim

        self.e_embeddings = nn.Embedding(self.num_entities, embed_dim)
        self.r_embeddings = nn.Embedding(self.num_relations, embed_dim)

        nn.init.xavier_uniform_(self.e_embeddings.weight)
        nn.init.xavier_uniform_(self.r_embeddings.weight)

    def forward(self, batch):
        # input:  (s, r, o, s', o') * B
        # output:  embeddings of (s, r, o, s', o') * B * N
        outputs = torch.stack([
            self.e_embeddings(batch["s"]),
            self.r_embeddings(batch["r"]),
            self.e_embeddings(batch["o"]),
            self.e_embeddings(batch["s'"]),
            self.e_embeddings(batch["o'"]),
        ])
        outputs = F.normalize(outputs, p=2, dim=-1)
        return outputs
