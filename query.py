from scipy.spatial import KDTree
from kg_data import FB5KDataset
from model import TransEModel, SavableModel
from tqdm import tqdm


def kg_completion(kg: FB5KDataset, triplets, e_embeddings, r_embeddings):
    print('Building KD Tree')
    kdtree = KDTree(e_embeddings)
    print('Done')

    hits = 0
    for (s, r, o_true) in triplets:
        s_vec = e_embeddings[kg.e2id[s]]
        r_vec = e_embeddings[kg.r2id[r]]
        o_true_idx = kg.e2id[o_true]
        pred_vec = s_vec + r_vec
        _, top_k_indices = kdtree.query(pred_vec, k=10, p=1)
        print(top_k_indices)
        if o_true_idx in top_k_indices:
            hits += 1
        print(s, r, o_true)
        print(s, r, kg.id2e[top_k_indices[0]])
        print('')
    return hits / len(triplets)


def eval_kg_completion(checkpoint):
    kg = FB5KDataset.get_instance()
    model: TransEModel = TransEModel(num_entities=len(kg.e2id), num_relations=len(kg.r2id), embed_dim=30)
    model.load(checkpoint)

    e_embeddings = model.export_entity_embeddings()
    r_embeddings = model.export_relation_embeddings()

    hits = kg_completion(kg, kg.triplets[:100], e_embeddings, r_embeddings)
    print("Hits@10", hits)


if __name__ == '__main__':
    eval_kg_completion('checkpoints/trans-e-best.pt')
