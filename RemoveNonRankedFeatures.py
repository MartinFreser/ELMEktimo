__author__ = 'Martin'
import numpy as np
class RemoveNonRankedFeatures():
    def __init__(self):
        pass
    def fit(self,X):
        self.nonRankedFeats = np.any(np.logical_or(X>1, X<0), axis=0)
    def transform(self, X):
        return X[:,~self.nonRankedFeats]
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)