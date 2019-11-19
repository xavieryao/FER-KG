import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from kg_data import FB5KDataset, FilteredFB5KDataset
from model import SavableModel, TransEModel


class EmbRegressionModel(SavableModel):
    def __init__(self, embed_dim, num_contexts, context_length, num_layers):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_contexts = num_contexts
        self.context_length = context_length
        self.num_layers = num_layers

        self.context_encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=2, dim_feedforward=128)
        self.context_encoder = nn.TransformerEncoder(self.context_encoder_layer, num_layers)

        self.context_pool = nn.MaxPool1d(context_length + 1)

        self.shot_encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=2, dim_feedforward=128)
        self.shot_encoder = nn.TransformerEncoder(self.context_encoder_layer, num_layers)

        self.output = nn.Linear(embed_dim * num_contexts, embed_dim)

    def forward(self, x: torch.Tensor):
        # Input: B x C x L x N
        # Output: N x L
        batch_size, num_contexts, context_length, num_embeddings = x.shape

        # context encoding
        context_encodings = []
        for c in range(num_contexts):
            ctx = x[:, c].transpose(0, 1)  # L x B x N
            ctx: torch.Tensor = self.context_encoder(ctx)  # L x B x N
            ctx = ctx.permute(1, 2, 0)  # B x N x L
            ctx = self.context_pool(ctx).squeeze()  # B x N x 1
            context_encodings.append(ctx)
        context_encodings = torch.stack(context_encodings)  # C x B x N

        shot_encoding = self.shot_encoder(context_encodings)  # C x B x N
        shot_encoding = shot_encoding.transpose(0, 1)  # B x C x N
        shot_encoding = shot_encoding.reshape(batch_size, -1)  # B x N

        output = self.output(shot_encoding)
        return F.normalize(output, p=2, dim=-1)


def train(model: EmbRegressionModel):
    train_writer = SummaryWriter('runs/FSReg_train')
    val_writer = SummaryWriter('runs/FSReg_val')

    kg = FB5KDataset.get_instance()
    filtered_dataset = FilteredFB5KDataset(kg, min_entity_freq=0.8, min_relation_freq=0.5)
    trans_e_model = TransEModel(len(kg.e2id), len(kg.r2id), 50)
    trans_e_model.load('checkpoints/trans-e-10.pt')
    e_embeddings = trans_e_model.e_embeddings.weight
    r_embeddings = trans_e_model.r_embeddings.weight

    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters())

    try:
        os.mkdir('checkpoints')
        os.mkdir('runs')
    except FileExistsError:
        pass

    for epoch in range(10):
        data_generator = filtered_dataset.get_batch_generator(
            batch_size=16,
            emb_dim=50,
            e_embeddings=e_embeddings,
            r_embeddings=r_embeddings,
            num_context=4,
            length=2
        )
        running_loss = 0.0
        for i, (batch_X, batch_Y) in enumerate(data_generator):
            optimizer.zero_grad()

            Y_pred = model(batch_X)
            loss = criterion(batch_Y, Y_pred)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d]     loss: %.6f' %
                      (epoch + 1, i + 1, running_loss / 100))
                steps = 1  # FIXME
                train_writer.add_scalar('epoch', epoch + 1, steps)
                train_writer.add_scalar('loss', running_loss / 100, steps)

                running_loss = 0.0

            # TODO: validate
            # TODO: save


def main():
    model = EmbRegressionModel(
        embed_dim=50,
        num_contexts=4,
        context_length=4,
        num_layers=1
    )
    train(model)


if __name__ == '__main__':
    main()
