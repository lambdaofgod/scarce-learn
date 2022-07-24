import attr
import itertools
from frozendict import frozendict
import tqdm
from sklearn import metrics
from scarce_learn import zero_shot


@attr.s
class HyperParamOptimizer:

    learner: zero_shot.ZeroShotClassifier = attr.ib()
    hyperparams = attr.ib()

    def fit(
        self,
        X_train,
        y_train,
        class_attributes_train,
        X_val,
        y_val,
        class_attributes_val,
    ):
        hyperparam_scores = {}
        models = {}
        n_hyperparam_values = np.product(
            [len(vals) for vals in self.hyperparams.values()]
        )
        for hyperparam_dict in tqdm.tqdm(
            self.get_hyperparam_dicts(hyperparams), total=n_hyperparam_values
        ):
            model = self.learner.__class__(**hyperparam_dict)
            models[hyperparam_dict] = model
            model.fit(X_train, y_train, class_attributes_train)
            hyperparam_scores[hyperparam_dict] = {}
            hyperparam_scores[hyperparam_dict]["train"] = model.score(
                X_train, y_train, class_attributes_train
            )
            hyperparam_scores[hyperparam_dict]["val"] = model.score(
                X_val, y_val, class_attributes_val
            )
        self.hyperparam_scores = hyperparam_scores
        self.models = models

    def get_best_hyperparams(self, split="val"):
        return max(
            self.hyperparam_scores.keys(),
            key=lambda k: self.hyperparam_scores[k][split],
        )

    def get_best_model(self, split="val"):
        return self.models[self.get_best_hyperparams(split=split)]

    def best_score(self, split="val"):
        return self.hyperparam_scores[self.get_best_hyperparams(split)][split]

    def score(
        self,
        X,
        y,
        class_attributes,
        labels_to_attributes=None,
        metric=metrics.accuracy_score,
    ):
        return self.get_best_model().score(
            X, y, class_attributes, labels_to_attributes, metric=metrics.accuracy_score
        )

    @classmethod
    def get_hyperparam_dicts(cls, hyperparams):
        hyperparam_names = hyperparams.keys()
        for hyperparam_values in itertools.product(*hyperparams.values()):
            hyperparam_values
            yield frozendict(dict(zip(hyperparam_names, hyperparam_values)))
