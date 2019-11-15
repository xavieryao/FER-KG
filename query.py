from scipy.spatial import KDTree
from kg_data import FB5KDataset
from model import TransEModel, SavableModel
from tqdm import tqdm
import random


def kg_completion(kg: FB5KDataset, triplets, e_embeddings, r_embeddings):
    print('Building KD Tree')
    kdtree = KDTree(e_embeddings)
    print('Done')

    hits = 0
    n = 0
    for (s, r, o_true) in triplets:
        n += 1
        s_vec = e_embeddings[kg.e2id[s]]
        r_vec = e_embeddings[kg.r2id[r]]
        o_true_idx = kg.e2id[o_true]
        pred_vec = s_vec + r_vec
        _, top_k_indices = kdtree.query([pred_vec], k=10, p=1)
        top_k_indices = top_k_indices[0]
        if o_true_idx in top_k_indices:
            hits += 1
        print(hits / n)
    return hits / len(triplets)


def eval_kg_completion(checkpoint):
    kg = FB5KDataset.get_instance()
    model: TransEModel = TransEModel(num_entities=len(kg.e2id), num_relations=len(kg.r2id), embed_dim=50)
    model.load(checkpoint)

    e_embeddings = model.export_entity_embeddings()
    r_embeddings = model.export_relation_embeddings()

    triplets = kg.valid_triplets
    random.shuffle(triplets)
    triplets = [x for x in triplets if x[0] != '<UNK>' and x[1] != '<UNK>' and x[2] != '<UNK>']

    hits = kg_completion(kg, triplets, e_embeddings, r_embeddings)
    print("Hits@10", hits)


if __name__ == '__main__':
    eval_kg_completion('checkpoints/trans-e-best.pt')
