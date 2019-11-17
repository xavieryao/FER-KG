import random
from collections import Counter, defaultdict

import torch


class FB5KDataset:
    _instance = None

    @classmethod
    def get_instance(cls):
        if not cls._instance:
            cls._instance = FB5KDataset()
        return cls._instance

    def __init__(self):
        # read data
        with open('../FB15K/train.txt') as f:
            self.triplets = []
            for line in f:
                self.triplets.append(tuple(line[:-1].split('\t')))

        # count entities and relations
        entity_counter = Counter()
        relation_counter = Counter()
        for (s, r, o) in self.triplets:
            entity_counter[s] += 1
            entity_counter[o] += 1
            relation_counter[r] += 1
        self.entity_counter = entity_counter
        self.relation_counter = relation_counter

        # create mappings
        self.e2id = {}
        self.id2e = {}
        self.r2id = {}
        self.id2r = {}
        for i, (e, _) in enumerate(entity_counter.most_common()):
            self.e2id[e] = i
            self.id2e[i] = e
        unk_idx = len(self.e2id)
        self.e2id['<UNK>'] = unk_idx
        self.id2e[unk_idx] = '<UNK>'
        for i, (r, _) in enumerate(relation_counter.most_common()):
            self.r2id[r] = i
            self.id2r[i] = r
        unk_idx = len(self.r2id)
        self.r2id['<UNK>'] = unk_idx
        self.id2r[unk_idx] = '<UNK>'

        # placeholders for other fields
        self._valid_triplets = None
        self._test_triplets = None

    @property
    def valid_triplets(self):
        if self._valid_triplets is None:
            self._valid_triplets = []
            with open('../FB15K/valid.txt') as f:
                for line in f:
                    self._valid_triplets.append(tuple(line[:-1].split('\t')))
        return self._valid_triplets

    @property
    def test_triplets(self):
        if self._test_triplets is None:
            self._test_triplets = []
            with open('../FB15K/test.txt') as f:
                for line in f:
                    self._test_triplets.append(tuple(line[:-1].split('\t')))
        return self._test_triplets

    @staticmethod
    def _get_new_triplet(corruption, old_triplet, all_entities):
        s, r, o = old_triplet
        new_e = random.choice(all_entities)
        if corruption == 's':
            return new_e, r, o
        else:
            return s, r, new_e

    def convert_triplets_to_batch(self, triplets):
        batch = {
            's': [],
            'r': [],
            'o': []
        }
        for s, r, o in triplets:
            batch['s'].append(self.e2id.get(s, self.e2id['<UNK>']))
            batch['r'].append(self.r2id.get(r, self.r2id['<UNK>']))
            batch['o'].append(self.e2id.get(o, self.e2id['<UNK>']))
        for k, v in batch.items():
            batch[k] = torch.LongTensor(v)
        return batch

    def get_batch_generator(self, batch_size, shuffle=True):
        all_triplets_set = set(self.triplets)
        all_triplets = list(all_triplets_set)
        all_entities = list(self.e2id.keys())

        if shuffle:
            random.shuffle(all_triplets)
        num_batches = (len(all_triplets) + batch_size - 1) // batch_size
        for i in range(num_batches):
            batch_triplets = all_triplets[i * batch_size: (i + 1) * batch_size]
            neg_triplets = []
            pos_triplets = []
            for (s, r, o) in batch_triplets:
                corruption = random.choice(['s', 'o'])
                new_triplet = self._get_new_triplet(corruption, (s, r, o), all_entities)
                retry = 5
                while retry > 0 and new_triplet in all_triplets_set:
                    retry -= 1
                    new_triplet = self._get_new_triplet(corruption, (s, r, o), all_entities)
                if new_triplet in all_triplets_set:
                    continue
                pos_triplets.append((s, r, o))
                neg_triplets.append(new_triplet)
            yield pos_triplets, neg_triplets


class FilteredFB5KDataset:
    def __init__(self, kg: FB5KDataset, min_entity_freq=0., max_entity_freq=1., min_relation_freq=0., max_relation_freq=1.):
        self.kg = kg

        self.e2id = kg.e2id
        self.id2e = kg.id2e
        self.r2id = kg.r2id
        self.id2r = kg.id2r

        num_entities = int(len(kg.entity_counter))
        num_relations = int(len(kg.relation_counter))
        self.entities = kg.entity_counter.most_common()[
                        int(num_entities * (1 - max_entity_freq)): int(num_entities * (1 - min_entity_freq))]
        self.relations = kg.relation_counter.most_common()[
                        int(num_relations * (1 - max_relation_freq)): int(num_relations * (1 - min_relation_freq))]

        entity_set = set(x[0] for x in self.entities)
        relation_set = set(x[0] for x in self.relations)

        self.triplets = []
        for s, r, o in kg.triplets:
            if s in entity_set and r in relation_set and o in entity_set:
                self.triplets.append((s, r, o))
        print('# Entity', len(entity_set))
        print('# Relation', len(relation_set))
        print('# Triplet', len(self.triplets))

        # train-validation-test split
        rnd = random.Random(42)
        rnd.shuffle(self.triplets)
        self.train_triplets = self.triplets[:int(0.8 * len(self.triplets))]
        self.validation_triplets = self.triplets[int(0.8 * len(self.triplets)):]

        # build index
        self.train_head_to_triplets, self.train_tail_to_triplets = self._build_index(self.train_triplets)
        self.validation_head_to_triplets, self.validation_tail_to_triplets = self._build_index(self.validation_triplets)

    @staticmethod
    def _build_index(triplets):
        head_to_triplets = defaultdict(list)
        tail_to_triplets = defaultdict(list)
        for s, r, o in triplets:
            head_to_triplets[s].append((s, r, o))
            tail_to_triplets[o].append((s, r, o))
        return head_to_triplets, tail_to_triplets

    @staticmethod
    def sample_relational_path(entity, length, head_to_triplets, tail_to_triplets):
        forward_path = [('e', entity)]
        backward_path = [('e', entity)]
        for i in range(length):
            triplets = head_to_triplets[forward_path[-1][1]]
            if len(triplets) == 0:
                break
            next_triplet = random.choice(triplets)
            forward_path.append(('r', next_triplet[1]))
            forward_path.append(('e', next_triplet[2]))
        for i in range(length):
            triplets = tail_to_triplets[forward_path[-1][1]]
            if len(triplets) == 0:
                break
            next_triplet = random.choice(triplets)
            forward_path.append(('r', next_triplet[1]))
            forward_path.append(('e', next_triplet[0]))
        return forward_path, backward_path

    def get_batch_generator(self, batch_size, e_embeddings, r_embeddings, length, shuffle=True):
        all_triplets = self.train_triplets
        # build index

        if shuffle:
            random.shuffle(all_triplets)
        num_batches = (len(all_triplets) + batch_size - 1) // batch_size
        for i in range(num_batches):
            batch_triplets = all_triplets[i * batch_size: (i + 1) * batch_size]


def main():
    kg = FB5KDataset.get_instance()
    filtered_dataset = FilteredFB5KDataset(kg, min_entity_freq=0.8, min_relation_freq=0.5)
    for tp in filtered_dataset.train_triplets[:10]:
        forward_path, backward_path = FilteredFB5KDataset.sample_relational_path(tp[0], 2, filtered_dataset.train_head_to_triplets, filtered_dataset.train_tail_to_triplets)
        print("FP", forward_path)
        print("BP", backward_path)


if __name__ == '__main__':
    main()