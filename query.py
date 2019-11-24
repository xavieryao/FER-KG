from kg_data import FB5KDataset
from model import TransEModel
import random
import torch
from tqdm import tqdm


def scoring(kg: FB5KDataset, triplets, e_embeddings, r_embeddings):
    e_embeddings = torch.Tensor(e_embeddings)
    r_embeddings = torch.Tensor(r_embeddings)
    scores = []
    for (s, r, o) in tqdm(triplets):
        s_vec = torch.Tensor(e_embeddings[kg.e2id.get(s, kg.e2id['<UNK>'])])
        r_vec = torch.Tensor(r_embeddings[kg.r2id.get(r, kg.r2id['<UNK>'])])
        o_vec = torch.Tensor(e_embeddings[kg.e2id.get(o, kg.e2id['<UNK>'])])
        pred_score = torch.norm(s_vec + r_vec - o_vec, p=1, dim=-1).item()
        scores.append(pred_score)
    return sum(scores) / len(scores)


def kg_completion(kg: FB5KDataset, triplets, e_embeddings, r_embeddings):
    e_embeddings = torch.Tensor(e_embeddings)
    r_embeddings = torch.Tensor(r_embeddings)
    hits = 0
    n = 0
    rank = []
    for (s, r, o_true) in tqdm(triplets):
        n += 1
        s_vec = torch.Tensor(e_embeddings[kg.e2id.get(s, kg.e2id['<UNK>'])])
        r_vec = torch.Tensor(r_embeddings[kg.r2id.get(r, kg.r2id['<UNK>'])])
        o_true_idx = kg.e2id.get(o_true, kg.e2id['<UNK>'])
        pred_scores = torch.norm(s_vec + r_vec - e_embeddings, p=1, dim=-1)
        ranking = torch.argsort(pred_scores).tolist()
        top_k_indices = ranking[:10]
        if o_true_idx in top_k_indices:
            hits += 1
        rank.append(ranking.index(o_true_idx))

        # print('hits', hits/n)
    return hits / len(triplets), sum(rank) / len(triplets)


def eval_kg_completion(checkpoint, ds='validation'):
    kg = FB5KDataset.get_instance()
    model: TransEModel = TransEModel(num_entities=len(kg.e2id), num_relations=len(kg.r2id), embed_dim=50)
    model.load(checkpoint)

    e_embeddings = model.export_entity_embeddings()
    r_embeddings = model.export_relation_embeddings()

    if ds == 'validation':
        triplets = kg.valid_triplets[:500]
    else:
        triplets = kg.test_triplets
    triplets = [x for x in triplets if x[0] != '<UNK>' and x[1] != '<UNK>' and x[2] != '<UNK>']

    #hits = kg_completion(kg, triplets, e_embeddings, r_embeddings)
    #print("Hits@10", hits)
    score = scoring(kg, triplets, e_embeddings, r_embeddings)
    print('Test score', score)


if __name__ == '__main__':
    eval_kg_completion('checkpoints/trans-e-best.pt', 'test')