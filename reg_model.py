import os
import random
import sys
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from kg_data import FB5KDataset, FilteredFB5KDataset
from model import SavableModel, TransEModel
from query import kg_completion, scoring
from config import load_config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len = 10):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len * 2
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(self.seq_len, d_model, device=device, requires_grad=False)
        for pos in range(self.seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        x = x + self.pe[:,:self.seq_len]
        return x


class EmbRegressionModel(SavableModel):
    def __init__(self, embed_dim, num_contexts, context_length, num_layers):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_contexts = num_contexts
        self.context_length = context_length
        self.num_layers = num_layers

        if config['positional_encoding']:
            self.positional_encoding = PositionalEncoding(self.embed_dim, self.context_length)

        self.context_encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=config['nhead'], dim_feedforward=config['dim_feedforward'], dropout=config['dropout'])
        self.context_encoder_layer_norm = nn.LayerNorm([embed_dim])
        self.context_encoder = nn.TransformerEncoder(self.context_encoder_layer, num_layers, norm=self.context_encoder_layer_norm)

        self.context_pool = nn.MaxPool1d(context_length + 1)

        self.shot_encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=config['nhead'], dim_feedforward=config['dim_feedforward'], dropout=config['dropout'])
        self.shot_encoder_layer_norm = nn.LayerNorm([embed_dim])
        self.shot_encoder = nn.TransformerEncoder(self.context_encoder_layer, num_layers, norm=self.shot_encoder_layer_norm)

        self.output = nn.Linear(embed_dim * num_contexts, embed_dim)

    def forward(self, x: torch.Tensor):
        # Input: B x C x L x N
        # Output: N x L
        batch_size, num_contexts, context_length, num_embeddings = x.shape

        if config['positional_encoding']:
            x = self.positional_encoding(x)

        # context encoding
        context_encodings = []
        for c in range(num_contexts):
            ctx = x[:, c].transpose(0, 1)  # L x B x N

            ctx: torch.Tensor = self.context_encoder(ctx)  # L x B x N
            ctx = ctx.permute(1, 2, 0)  # B x N x L
            ctx = self.context_pool(ctx).squeeze()  # B x N
            context_encodings.append(ctx)
        context_encodings = torch.stack(context_encodings)  # C x B x N

        shot_encoding = self.shot_encoder(context_encodings)  # C x B x N
        shot_encoding = shot_encoding.transpose(0, 1)  # B x C x N
        shot_encoding = shot_encoding.reshape(batch_size, -1)  # B x N

        output = self.output(shot_encoding)
        return F.normalize(output, p=2, dim=-1)


def validate(model: EmbRegressionModel, xs, ys_true):
    xs = xs.to(device)
    ys_true = ys_true.to(device)
    criterion = nn.L1Loss().to(device)
    ys_pred = model(xs)
    return criterion(ys_pred, ys_true).cpu()


def train(model: EmbRegressionModel):
    train_writer = SummaryWriter(f"runs/FSReg_{config['name']}_train")
    val_writer = SummaryWriter(f"runs/FSReg_{config['name']}_val")

    kg = FB5KDataset.get_instance()
    filtered_dataset = FilteredFB5KDataset(kg, min_entity_freq=config['min_entity_freq'], min_relation_freq=0.5)
    trans_e_model = TransEModel(len(kg.e2id), len(kg.r2id), config['embed_dim'])
    trans_e_model.load(config['trans-e-model'])
    e_embeddings = trans_e_model.export_entity_embeddings()
    r_embeddings = trans_e_model.export_relation_embeddings()

    valid_X, valid_Y = filtered_dataset.get_valid_data(
        emb_dim=config['embed_dim'],
        e_embeddings=e_embeddings,
        r_embeddings=r_embeddings,
        num_context=config['num_context'],
        length=config['length']
    )

    criterion = nn.L1Loss().to(device)
    optimizer = Adam(model.parameters())

    try:
        os.mkdir('checkpoints')
        os.mkdir('runs')
    except FileExistsError:
        pass

    steps = 0
    best_val_loss = float('+inf')
    best_hits = 0
    best_mean_rank = float('+inf')
    for epoch in range(30000):
        data_generator = filtered_dataset.get_train_batch_generator(
            batch_size=config['batch_size'],
            emb_dim=config['embed_dim'],
            e_embeddings=e_embeddings,
            r_embeddings=r_embeddings,
            num_context=config['num_context'],
            length=config['length']
        )
        running_loss = 0.0
        for i, (batch_X, batch_Y) in enumerate(data_generator):
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

            steps += len(batch_X)
            optimizer.zero_grad()

            model.train()
            Y_pred = model(batch_X)
            loss = criterion(batch_Y, Y_pred)
            loss.backward()
            optimizer.step()

            running_loss += loss.cpu().item()

            TRAIN_REPORT_FREQ = 5
            VAL_REPORT_FREQ = 100
            if i % TRAIN_REPORT_FREQ == TRAIN_REPORT_FREQ - 1:
                print('[%d, %5d]     loss: %.6f' %
                      (epoch + 1, i + 1, running_loss / TRAIN_REPORT_FREQ))
                train_writer.add_scalar('epoch', epoch + 1, steps)
                train_writer.add_scalar('loss', running_loss / TRAIN_REPORT_FREQ, steps)

                running_loss = 0.0
        # validate after each epoch
        model.eval()
        val_loss = validate(model, valid_X, valid_Y)
        print('[%d]     val loss: %.6f' %
              (epoch + 1,  val_loss))
        val_writer.add_scalar('loss', val_loss, steps)

        model.save(f"checkpoints/reg_{config['name']}_last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(f"checkpoints/reg_{config['name']}_best.pt")
        

        if epoch % 10 == 9:
            hits, m_rk = external_validation(model, filtered_dataset)
            val_writer.add_scalar('hits', hits, steps)
            val_writer.add_scalar('mean_rank', m_rk, steps)

            if hits > best_hits:
                best_hits = hits
                model.save(f"checkpoints/reg_{config['name']}_best_hits.pt")
            if m_rk < best_mean_rank:
                best_mean_rank = m_rk
                model.save(f"checkpoints/reg_{config['name']}_best_mean_rank.pt")


def external_validation(model, filtered_dataset, top=500):
    kg = FB5KDataset.get_instance()
    test_entities = list(set(kg.e2id.keys()) - set(x[0] for x in filtered_dataset.entities))
    trans_e_model = TransEModel(len(kg.e2id), len(kg.r2id), config['embed_dim'])
    trans_e_model.load(config['trans-e-model'])
    e_embeddings = trans_e_model.export_entity_embeddings()
    r_embeddings = trans_e_model.export_relation_embeddings()

    test_Xs, _ = next(filtered_dataset.get_batch_generator(
        entities=test_entities,
        batch_size=len(test_entities),
        emb_dim=config['embed_dim'],
        e_embeddings=e_embeddings,
        r_embeddings=r_embeddings,
        num_context=config['num_context'],
        length=config['length'],
        shuffle=False
    ))

    triplets = filtered_dataset.low_freq_triplets[:top]
    return external_eval(model, test_entities, test_Xs, triplets)


def external_eval(model, entities, test_Xs, triplets):
    kg = FB5KDataset.get_instance()
    trans_e_model = TransEModel(len(kg.e2id), len(kg.r2id), config['embed_dim'])
    trans_e_model.load(config['trans-e-model'])
    e_embeddings = trans_e_model.export_entity_embeddings()
    r_embeddings = trans_e_model.export_relation_embeddings()


    test_Ys = model(test_Xs.to(device)).detach().cpu().numpy()
    new_e_embeddings = e_embeddings.copy()
    repl = 0
    for i, et in enumerate(entities):
        if et not in kg.e2id:
            continue
        et_id = kg.e2id[et]
        new_e_embeddings[et_id] = test_Ys[i]
        repl += 1

    print(f'replaced {repl} out of {len(entities)} embeddings')
    # evaluate kg completion
    triplets = [x for x in triplets if x[0] != '<UNK>' and x[1] != '<UNK>' and x[2] != '<UNK>']

    score = scoring(kg, triplets, new_e_embeddings, r_embeddings)
    print("Score", score)

    hits = kg_completion(kg, triplets, new_e_embeddings, r_embeddings)
    print("Hits@10", hits[0], "meanRank", hits[1])
    return hits


def test(model, filtered_dataset):
    kg = FB5KDataset.get_instance()
    trans_e_model = TransEModel(len(kg.e2id), len(kg.r2id), config['embed_dim'])
    trans_e_model.load(config['trans-e-model'])
    e_embeddings = trans_e_model.export_entity_embeddings()
    r_embeddings = trans_e_model.export_relation_embeddings()
    test_Xs = filtered_dataset.get_test_data(
        emb_dim=config['embed_dim'],
        e_embeddings=e_embeddings,
        r_embeddings=r_embeddings,
        num_context=config['num_context'],
        length=config['length']
    )
    return external_eval(
        model=model,
        entities=filtered_dataset.test_entities,
        test_Xs=test_Xs,
        triplets=kg.test_triplets
    )


def main():
    model = EmbRegressionModel(
        embed_dim=config['embed_dim'],
        num_contexts=config['num_context'],
        context_length=config['length'] * 2,
        num_layers=config['layers']
    )
    model = model.to(device)
    if sys.argv[1] == 'train':
        train(model)
    elif sys.argv[1] == 'test':
        model.load(sys.argv[3])
        model.eval()
        kg = FB5KDataset.get_instance()
        filtered_dataset = FilteredFB5KDataset(kg, min_entity_freq=config['min_entity_freq'], min_relation_freq=0.5)
        test(model, filtered_dataset)


if __name__ == '__main__':
    config = load_config('config.json', sys.argv[2])
    main()
