__author__ = 'Martin'
import numpy as np
"""
    Razred, ki z metodo fit() poisce featurje, ki imajo vrednosti vecjo od 1 ali manjse od 0 (Torej vrednosti niso
    rangirane), si zapomni indekse in s metodo transform() vrne featurje, kateri so vsi rangirani
"""
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