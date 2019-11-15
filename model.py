from torch import nn
import torch
from torch.nn import functional as F
from kg_data import FB5KDataset
import random


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
        s, o = self.e_embeddings(batch['s']), self.e_embeddings(batch['o'])
        r = self.r_embeddings(batch['r'])
        s, r, o = [F.normalize(x, p=2, dim=-1) for x in (s, r, o)]
        return F.normalize(s + r - o, 1, -1)


def train(model):
    dataset = FB5KDataset()
