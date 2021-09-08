import attr
from toolz import partial

import torch
import ignite
from torch import nn, optim
from ignite.contrib.handlers import tqdm_logger


def bilinear_feature_similarity(weights, embedding, class_features):
    """
    embedding * W * class_features
    """
    return torch.matmul(embedding, torch.matmul(weights, class_features.T))


def similarity_based_hinge_loss(weights, embedding, label, label_embeddings, feature_similarity=bilinear_feature_similarity):
    """
    see https://arxiv.org/pdf/1703.04394.pdf
    equations (4) and (7) only differn in final per-class aggregation
    this function computes value before this final aggregation step
    """
    indicator = torch.ones(label_embeddings.shape[0], dtype=bool)
    indicator[label] = 0
    correct_class_similarity = feature_similarity(weights, embedding, label_embeddings[label])
    wrong_class_similarities = feature_similarity(weights, embedding, label_embeddings[indicator])
    return - correct_class_similarity + wrong_class_similarities


def get_dataloader(X, y, batch_size=16):
    ds = torch.utils.data.TensorDataset(torch.Tensor(X), torch.Tensor(y).int())
    return torch.utils.data.DataLoader(ds, batch_size=batch_size)


def process_function(engine, batch, loss_fn, optimizer, y_features, use_cuda=True):
    optimizer.zero_grad()
    x, y = batch
    if use_cuda:
        x = x.cuda()
        y = y.cuda()
    loss = loss_fn(x, y, y_features)
    y_pred = loss_fn.predict(x, y_features)
    loss.backward()
    optimizer.step()
    return y_pred, y, {'loss': loss.item()}


def run_training_loop(loss_fn, train_dataloader, epochs, y_features_train, optimizer=optim.Adagrad, use_cuda=True):
    if use_cuda:
        y_features_train = y_features_train.cuda()
    engine_fn = partial(process_function, loss_fn=loss_fn, optimizer=optimizer(loss_fn.parameters()), y_features=y_features_train)
    trainer = ignite.engine.Engine(engine_fn)
    pbar = tqdm_logger.ProgressBar()
    pbar.attach(trainer)
    trainer.run(train_dataloader, max_epochs=epochs)
    return loss_fn
