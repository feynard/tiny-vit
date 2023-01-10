import torch
import tqdm

import yaml
import sys
from types import SimpleNamespace

from dataset import create_loaders
from vit import VisionTransformer


def read_config(config_file: str):
    with open(config_file, 'r') as f:
        return SimpleNamespace(**yaml.safe_load(f))


def run_epoch(model, optimizer, loss_function, train_dataloader, validation_dataloader, use_gpu) -> dict:

    train_loss = []
    model.train()
    for x, y in train_dataloader:
        optimizer.zero_grad()
        y_pred = model(x.cuda() if use_gpu else x)
        loss = loss_function(y_pred, y.cuda() if use_gpu else y)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    validation_loss = []
    model.eval()
    with torch.no_grad():
        for x, y in validation_dataloader:
            y_pred = model(x.cuda() if use_gpu else x)
            loss = loss_function(y_pred, y.cuda() if use_gpu else y)

            validation_loss.append(loss.item())

    return {
        'train': torch.tensor(train_loss),
        'validation': torch.tensor(validation_loss)
    }


if __name__ == '__main__':

    config = read_config(sys.argv[1])

    train_dataloader, validation_dataloader = create_loaders(
        config.dataset,
        config.batch_size,
        config.train_split,
        config.flip_probability,
        config.rotation_angle
    )

    transformer = VisionTransformer(
        image_shape=config.image_shape,
        patch_size=config.patch_size,
        num_layers=config.num_layers,
        transformer_dim=config.transformer_dim,
        attention_heads=config.attention_heads,
        num_classes=config.num_classes,
        dropout=config.dropout,
        pooling_type=config.pooling_type
    )

    if config.use_gpu:
        transformer = transformer.cuda()

    optimizer = torch.optim.Adam(transformer.parameters(), lr=config.lr)
    loss_function = torch.nn.CrossEntropyLoss()

    t = tqdm.trange(config.epochs, leave=True)
    for i in t:
        stats = run_epoch(
            transformer,
            optimizer,
            loss_function,
            train_dataloader,
            validation_dataloader,
            config.use_gpu)

        t.set_description(f"Train: {stats['train'].mean():.2f}, Validation: {stats['validation'].mean():.2f}")
        t.refresh()

    torch.save(transformer.state_dict(), config.weights)