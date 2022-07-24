import numpy as np
import attr
from toolz import partial
from scarce_learn.zero_shot import zsl_base

from sklearn import preprocessing
import torch
from torch import nn, optim
from scarce_learn.zero_shot import torch_util


class DEVISELayer(nn.Module):
    def __init__(self, n_features, n_class_features, margin, init_weights_std=0.1):
        super(DEVISELayer, self).__init__()
        init_weights = init_weights_std * torch.randn(n_features, n_class_features)
        self.weights = nn.Parameter(data=init_weights.cuda())
        self.margin = margin

    def forward(self, X, y, label_embeddings):
        loss = torch.Tensor([0]).cuda()
        for i in range(X.shape[0]):
            loss += self._devise_loss(X[i], y[i], label_embeddings)
        return loss / X.shape[0]

    def _devise_loss(self, embedding, label, label_embeddings):
        indicator = torch.ones(label_embeddings.shape[0], dtype=bool)
        indicator[label] = 0
        per_class_loss = torch_util.similarity_based_hinge_loss(
            self.weights, embedding, label, label_embeddings
        )
        return nn.ReLU()(self.margin + per_class_loss).sum()

    def predict(self, X, label_embeddings):
        class_similarities = torch_util.bilinear_feature_similarity(
            self.weights, X, label_embeddings
        )
        return torch.argmax(class_similarities, axis=1)

    @property
    def device(self):
        return next(self.parameters()).device


@attr.s
class DEVISELearner(zsl_base.ZeroShotClassifier):
    """
    see https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41473.pdf
    for the paper
    """

    margin = attr.ib(default=0.1)

    def fit(self, X, y, class_attributes, n_epochs=1, eval_set=None, batch_size=32):
        self.loss_fn = DEVISELayer(X.shape[1], class_attributes.shape[1], self.margin)
        le = preprocessing.LabelEncoder()
        ohe = preprocessing.OneHotEncoder()
        y_labels_encoded = le.fit_transform(y)
        y_labels_ohe = ohe.fit_transform(y_labels_encoded.reshape(-1, 1)).toarray()
        M = np.random.randn
        class_attributes_t = torch.Tensor(class_attributes).float()
        train_dataloader = torch_util.get_dataloader(X, y, batch_size=batch_size)
        torch_util.run_training_loop(
            self.loss_fn,
            train_dataloader,
            n_epochs,
            torch.Tensor(class_attributes).float(),
        )
        return self

    def predict(self, X, class_attributes):
        X_tensor = torch.tensor(X).float().cuda()
        class_attributes_tensor = torch.Tensor(class_attributes).float().cuda()
        return (
            self.loss_fn.predict(X_tensor, class_attributes_tensor)
            .cpu()
            .detach()
            .numpy()
        )

    def predict_raw(self, X):
        return X @ self.loss_fn.weights.cpu().detach().numpy()

    def get_loss(self, X, class_attributes, y):
        device = self.loss_fn.device
        return self.loss_fn(
            torch.Tensor(X).to(device),
            torch.LongTensor(y).to(device),
            torch.Tensor(class_attributes).to(device),
        ).item()
