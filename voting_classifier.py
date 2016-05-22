import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin

class VotingClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):

    def __init__(self, estimators, voting='hard', weights=None):

        self.estimators = estimators
        self.voting = voting
        self.weights = weights

        if self.voting not in ('soft', 'hard'):
            raise ValueError("Voting must be 'soft' or 'hard'; got (voting=%r)"
                             % self.voting)

        if self.weights and len(self.weights) != len(self.estimators):
            raise ValueError('Number of classifiers and weights must be equal'
                             '; got %d weights, %d estimators'
                             % (len(self.weights), len(self.estimators)))

    def fit(self, X, y):
        raise NotImplementedError('All estimators need to be already fitted')

    def transform(self, X):
        if self.voting == 'soft':
            return self._collect_probas(X)
        else:
            return self._predict(X)

    def predict(self, X):
        if self.voting == 'soft':
            maj = np.argmax(self.predict_proba(X), axis=1)

        else:  # 'hard' voting
            predictions = self._predict(X)
            maj = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                                      axis=1,
                                      arr=predictions)

        return maj

    def _predict(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict(X) for clf in self.estimators]).T

    @property
    def predict_proba(self):
        if self.voting == 'hard':
            raise AttributeError("predict_proba is not available when"
                                 " voting=%r" % self.voting)
        return self._predict_proba

    def _predict_proba(self, X):
        """Predict class probabilities for X in 'soft' voting """
        avg = np.average(self._collect_probas(X), axis=0, weights=self.weights)
        return avg

    def _collect_probas(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict_proba(X) for clf in self.estimators])

    def get_params(self, deep=True):
        """Return estimator parameter names for GridSearch support"""
        if not deep:
            return super(VotingClassifier, self).get_params(deep=False)
        else:
            out = super(VotingClassifier, self).get_params(deep=False)

            named_estimators = {}
            for i in range(0, len(self.estimators)):
                named_estimators['cls_' + str(i)] = self.estimators[i]

            out.update(named_estimators.copy())
            for name, step in named_estimators.items():
                for key, value in step.get_params(deep=True).items():
                    out['%s__%s' % (name, key)] = value
            return out
