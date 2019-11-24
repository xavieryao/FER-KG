import random
random.seed(233)
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
    def __init__(self, kg: FB5KDataset, min_entity_freq=0., max_entity_freq=1., min_relation_freq=0.,
                 max_relation_freq=1.):
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

        self.low_freq_triplets = []
        for (s, r, o) in kg.triplets:
            if s not in entity_set or o not in entity_set:
                self.low_freq_triplets.append((s, r, o))
        print("# low", len(self.low_freq_triplets))
        random.shuffle(self.low_freq_triplets)

        self.triplets = kg.triplets
        print('# Entity', len(entity_set))
        print('# Relation', len(relation_set))
        print('# Triplet', len(self.triplets))

        # train-validation-test split
        rnd = random.Random(42)
        rnd.shuffle(self.triplets)
        self.train_entities = [x[0] for x in self.entities[:int(0.8 * len(self.entities))]]
        self.validation_entities = [x[0] for x in self.entities[int(0.8 * len(self.entities)):]]

        # test entities
        test_entity_set = set()
        for s, _, o in kg.test_triplets:
            test_entity_set.add(s)
            test_entity_set.add(o)
        self.test_entities = list(test_entity_set - entity_set)

        # build index
        self.head_to_triplets, self.tail_to_triplets = self._build_index(self.triplets)

    @staticmethod
    def _build_index(triplets):
        head_to_triplets = defaultdict(list)
        tail_to_triplets = defaultdict(list)
        for s, r, o in triplets:
            head_to_triplets[s].append((s, r, o))
            tail_to_triplets[o].append((s, r, o))
        return head_to_triplets, tail_to_triplets

    def sample_relational_path(self, entity, length, rnd=None):
        if rnd is None:
            rnd = random
        forward_path = [('e', entity)]
        backward_path = [('e', entity)]
        for i in range(length):
            triplets = self.head_to_triplets[forward_path[-1][1]]
            if len(triplets) == 0:
                break
            next_triplet = rnd.choice(triplets)
            forward_path.append(('r', next_triplet[1]))
            forward_path.append(('e', next_triplet[2]))
        for i in range(length):
            triplets = self.tail_to_triplets[backward_path[-1][1]]
            if len(triplets) == 0:
                break
            next_triplet = rnd.choice(triplets)
            backward_path.append(('r', next_triplet[1]))
            backward_path.append(('e', next_triplet[0]))
        return forward_path, backward_path

    def get_train_batch_generator(self, batch_size, emb_dim, e_embeddings, r_embeddings, num_context, length, shuffle=True):
        return self.get_batch_generator(self.train_entities, batch_size, emb_dim, e_embeddings, r_embeddings,
                                         num_context, length, shuffle)

    def get_valid_data(self, emb_dim, e_embeddings, r_embeddings, num_context, length):
        return next(
            self.get_batch_generator(self.validation_entities, len(self.validation_entities), emb_dim, e_embeddings,
                                      r_embeddings, num_context, length, shuffle=False))

    def get_test_data(self, emb_dim, e_embeddings, r_embeddings, num_context, length):
        return next(
            self.get_batch_generator(self.test_entities, len(self.test_entities), emb_dim, e_embeddings,
                                      r_embeddings, num_context, length, shuffle=False))[0]

    def get_batch_generator(self, entities, batch_size, emb_dim, e_embeddings, r_embeddings, num_context, length,
                             shuffle=True):
        if shuffle:
            random.shuffle(entities)
            rnd = random
        else:
            rnd = random.Random(42)
        num_batches = (len(entities) + batch_size - 1) // batch_size
        for i in range(num_batches):
            entities_in_batch = entities[i * batch_size: (i+1) * batch_size]
            batch_X = []
            batch_Y = []
            for entity in entities_in_batch:
                contexts = []
                y_true = torch.Tensor(e_embeddings[self.e2id.get(entity, self.e2id['<UNK>'])])
                for _ in range(num_context):
                    forward_path, backward_path = self.sample_relational_path(entity, length, rnd)
                    for _ in range(length - (len(forward_path) - 1) // 2):
                        forward_path.append(('PAD', ''))
                        forward_path.append(('PAD', ''))
                    for _ in range(length - (len(backward_path) - 1) // 2):
                        backward_path.append(('PAD', ''))
                        backward_path.append(('PAD', ''))
                    path = backward_path[:0:-1] + forward_path[1:]
                    assert len(path) == length * 4
                    embeddings_in_path = []
                    for t, name in path:
                        if t == 'r':
                            embeddings_in_path.append(torch.Tensor(r_embeddings[self.r2id.get(name, self.r2id['<UNK>'])]))
                        elif t == 'e':
                            embeddings_in_path.append(torch.Tensor(e_embeddings[self.e2id.get(name, self.e2id['<UNK>'])]))
                        elif t == 'PAD':
                            embeddings_in_path.append(torch.zeros((emb_dim,)))

                    contexts.append(torch.stack(embeddings_in_path))
                batch_X.append(torch.stack(contexts))
                batch_Y.append(y_true)
            yield torch.stack(batch_X), torch.stack(batch_Y)
