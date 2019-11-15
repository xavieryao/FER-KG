from collections import Counter
import random


class FB5KDataset:
    def __init__(self):
        # read data
        with open('../FB15K/train.txt') as f:
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

    @staticmethod
    def _get_new_triple(corruption, old_triple, all_entities):
        s, r, o = old_triple
        new_e = random.choice(all_entities)
        if corruption == 's':
            return new_e, r, o
        else:
            return s, r, new_e

    def get_batch_generator(self, batch_size, shuffle=True):
        all_triples_set = set(self.triples)
        all_triples = list(all_triples_set)
        all_entities = list(self.e2id.keys())

        if shuffle:
            random.shuffle(all_triples)
        num_batches = (len(all_triples) + batch_size - 1) // batch_size
        for i in range(num_batches):
            batch_triples = all_triples[i * batch_size: (i+1) * batch_size]
            neg_triples = []
            pos_triples = []
            for (s, r, o) in batch_triples:
                corruption = random.choice(['s', 'o'])
                new_triple = self._get_new_triple(corruption, (s, r, o), all_entities)
                retry = 5
                while retry > 0 and new_triple in all_triples_set:
                    retry -= 1
                    new_triple = self._get_new_triple(corruption, (s, r, o), all_entities)
                if new_triple in all_triples_set:
                    continue
                pos_triples.append((s, r, o))
                neg_triples.append(new_triple)
            yield pos_triples, neg_triples
