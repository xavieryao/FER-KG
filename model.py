import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD
from torch.utils.tensorboard import SummaryWriter

from kg_data import FB5KDataset


class SavableModel(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class MarginRankingLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, pos_scores, neg_scores):
        dist = self.margin + pos_scores - neg_scores
        return torch.mean(torch.max(torch.Tensor([0]), dist))


class TransEModel(SavableModel):
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

    def export_entity_embeddings(self):
        embs = F.normalize(self.e_embeddings.weight, p=2, dim=-1)
        return embs.detach().numpy()

    def export_relation_embeddings(self):
        embs = F.normalize(self.r_embeddings.weight, p=2, dim=-1)
        return embs.detach().numpy()


def validate(model: nn.Module):
    dataset = FB5KDataset.get_instance()
    batch = dataset.convert_triplets_to_batch(dataset.valid_triplets)
    scores = model(batch)
    score = torch.mean(scores).item()
    return score


def train(model: SavableModel):
    train_writer = SummaryWriter('runs/TransE_train')
    val_writer = SummaryWriter('runs/TransE_val')
    dataset = FB5KDataset.get_instance()

    criterion: nn.Module = MarginRankingLoss(margin=1.0)
    #  optimizer = Adam(model.parameters(), weight_decay=0.01)
    optimizer = SGD(model.parameters(), lr=0.01, weight_decay=0.01)

    try:
        os.mkdir('checkpoints')
        os.mkdir('runs')
    except FileExistsError:
        pass

    best_val_score = float('+inf')
    for epoch in range(10):
        data_generator = dataset.get_batch_generator(batch_size=16)
        running_loss = 0.0
        running_score = 0.0
        for i, (pos_triplets, neg_triplets) in enumerate(data_generator):
            pos_batch = dataset.convert_triplets_to_batch(pos_triplets)
            neg_batch = dataset.convert_triplets_to_batch(neg_triplets)

            optimizer.zero_grad()

            pos_scores = model(pos_batch)
            neg_scores = model(neg_batch)

            loss: torch.Tensor = criterion(pos_scores, neg_scores)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_score += torch.mean(pos_scores).item()
            if i % 100 == 99:
                print('[%d, %5d]     loss: %.6f     score: %.6f' %
                      (epoch + 1, i + 1, running_loss / 100, running_score / 100))
                steps = epoch * (len(dataset.triplets) + 15) // 16 + i
                train_writer.add_scalar('epoch', epoch + 1, steps)
                train_writer.add_scalar('loss', running_loss / 100, steps)
                train_writer.add_scalar('score', running_score / 100, steps)

                running_loss = 0.0
                running_score = 0.0

            if i % 1000 == 999:
                valid_score = validate(model)
                print('[%d, %5d]     validation score: %.6f' %
                      (epoch + 1, i + 1, valid_score))
                steps = epoch * (len(dataset.triplets) + 15) // 16 + i
                val_writer.add_scalar('score', valid_score, steps)

                if valid_score < best_val_score:
                    best_val_score = valid_score
                    model.save('checkpoints/trans-e-best.pt')


if __name__ == '__main__':
    def main():
        dataset = FB5KDataset.get_instance()
        model = TransEModel(num_entities=len(dataset.e2id), num_relations=len(dataset.r2id), embed_dim=50)
        train(model)


    main()
