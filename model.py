import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from kg_data import FB5KDataset


class MarginRankingLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, pos_scores, neg_scores):
        dist = self.margin + pos_scores - neg_scores
        return torch.sum(torch.max(torch.Tensor([0]), dist))


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
        return torch.norm(s + r - o, p=1, dim=-1)


def train(model: nn.Module):
    dataset = FB5KDataset()

    criterion: nn.Module = MarginRankingLoss(margin=1.0)
    optimizer = Adam(model.parameters())

    for epoch in range(10):
        data_generator = dataset.get_batch_generator(batch_size=16)
        running_loss = 0.0
        for i, (pos_triples, neg_triples) in enumerate(data_generator):
            pos_batch = dataset.convert_triples_to_batch(pos_triples)
            neg_batch = dataset.convert_triples_to_batch(neg_triples)

            optimizer.zero_grad()

            pos_scores = model(pos_batch)
            neg_scores = model(neg_batch)

            loss: torch.Tensor = criterion(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d]     loss: %.6f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0


if __name__ == '__main__':
    dataset = FB5KDataset()
    model = TransEModel(num_entities=len(dataset.e2id), num_relations=len(dataset.r2id), embed_dim=30)
    train(model)
