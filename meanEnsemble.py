__author__ = 'Martin'
import numpy as np
from sklearn.base import clone
"""
    Klasifikator, ki vzame osnovni klasifikator in nauci vec istih osnovnih klasifikatorjev, nato pa povpreci
    napoved vsakega izmed teh klasifikatorjev
"""
class MeanEnsemble():
    def __init__(self,base_estimator, n_estimators=10):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators

    def fit(self,X,y):
        self.classes_ = np.unique(y)

        estimators = []
        oobErrors = []
        for i in range(self.n_estimators):
            est = clone(self.base_estimator)
            est.fit(X, y)
            estimators.append(est)
        self.estimators = estimators

    def predict_proba(self,X):
        proba = np.zeros((X.shape[0], len(self.classes_)))
        for est in self.estimators:
            proba += est.predict_proba(X)

        proba = proba/len(self.estimators)
        return proba
    def predict(self,X):
        return self.classes_.take(np.argmax(self.predict_proba(X), axis=1),
                                  axis=0)