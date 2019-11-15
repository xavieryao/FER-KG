from collections import Counter


class FB5KDataset:
    def __init__(self):
        # read data
        with open('FB15K/train.txt') as f:
            self.triples = []
            for line in f:
                self.triples.append(tuple(line[:-1].split('\t')))

        # count entities and relations
        entity_counter = Counter()
        relation_counter = Counter()
        for (s, r, o) in self.triples:
            entity_counter[s] += 1
            entity_counter[o] += 1
            relation_counter[r] += 1

        # create mappings
        self.e2id = {}
        self.id2e = {}
        self.r2id = {}
        self.id2r = {}
        for i, (e, _) in enumerate(entity_counter.most_common()):
            self.e2id[e] = i
            self.id2e[i] = e
        for i, (r, _) in enumerate(relation_counter.most_common()):
            self.r2id[r] = i
            self.id2r[i] = r
