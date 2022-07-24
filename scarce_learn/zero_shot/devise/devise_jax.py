import tqdm
import attr
import numpy as np

from scarce_learn.zero_shot import torch_util, zsl_base
from jax.experimental import optimizers
import functools

import jax.numpy as jnp
import jax


def per_example_similarity_based_hinge_loss(
    weights, label_embeddings, embedding, label, margin
):
    similarities = embedding @ weights @ label_embeddings.T
    label_similarities = similarities[label]
    correct_class_mask = 1 - jax.nn.one_hot([label], label_embeddings.shape[0])
    per_class_loss = jax.nn.relu(margin - label_similarities + similarities)
    return (per_class_loss * correct_class_mask).sum()


def get_similarity_based_hinge_loss(label_embeddings, embeddings, labels, margin):
    def similarity_based_hinge_loss(weights):
        f_s = functools.partial(per_example_similarity_based_hinge_loss, margin=margin)
        f = jax.vmap(jax.jit(f_s), in_axes=(None, None, 0, 0))
        return f(weights, label_embeddings, embeddings, labels).mean()

    return similarity_based_hinge_loss


def train_devise(
    weights,
    X,
    y,
    label_embeddings,
    margin,
    n_epochs=2,
    learning_rate=0.001,
    batch_size=16,
):
    opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
    opt_state = opt_init(weights)
    loader = torch_util.get_dataloader(np.array(X), np.array(y), batch_size=batch_size)

    loss_fn = get_similarity_based_hinge_loss(label_embeddings, X, y, margin)
    gradient_fn = jax.jit(jax.grad(loss_fn))

    for i in tqdm.tqdm(range(n_epochs)):
        for (j, (embeddings_np, labels_np)) in enumerate(loader):
            embeddings, labels = jnp.array(embeddings_np), jnp.array(labels_np)
            grads = gradient_fn(weights)
            opt_state = opt_update(j, grads, opt_state)
            weights = get_params(opt_state)
    return weights


@attr.s
class DEVISELearner(zsl_base.ZeroShotClassifier):
    """
    see https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41473.pdf
    for the paper
    """

    margin = attr.ib(default=0.1)

    def fit(
        self, X, y, class_attributes, n_epochs=10, learning_rate=1e-3, batch_size=16
    ):
        weights = jnp.array(
            np.random.rand(X.shape[1], class_attributes.shape[1]) / np.sqrt(X.shape[1])
        )
        self.weights = train_devise(
            weights,
            X,
            y,
            class_attributes,
            self.margin,
            n_epochs,
            learning_rate,
            batch_size,
        )

    def predict(self, X, class_attributes):
        similarities = X @ self.weights @ class_attributes.T
        return np.array(similarities.argmax(axis=1), copy=False)

    def get_loss(self, X, class_attributes, y):
        return get_similarity_based_hinge_loss(class_attributes, X, y, self.margin)(
            self.weights
        ).item()

    def predict_raw(self, X):
        return np.array(X @ self.weights, copy=False)
