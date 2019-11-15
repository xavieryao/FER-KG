from collections import Counter
import torch
import random


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
            batch_triplets = all_triplets[i * batch_size: (i+1) * batch_size]
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

