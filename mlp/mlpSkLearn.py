__author__ = 'Martin'
import Orange
import numpy as np

class MLP:
    def __init__(self, layers, lambda_=1.0, dropout = None):
        self.layers = layers
        self.lambda_ = lambda_
        self.dropout = dropout
    def fit(self,X,Y):
        cls = Orange.classification.MLPLearner(layers = [X.shape[1]]+self.layers+[2], lambda_=self.lambda_, dropout=self.dropout)
        data = Orange.data.Table(X,Y)
        self.domain = data.domain
        self.model = cls(data)
    def predict_proba(self,X):
        X = np.array(X)
        data= Orange.data.Table(self.domain,X, np.zeros(X.shape[0]))
        return self.model(data, 1)

if __name__ == "__main__":
    X = np.random.rand(4,10)
    Y = np.array([[0],[1],[1],[0]])
    mlp = MLP([100,100])
    mlp.fit(X,Y)
    print(mlp.predict_proba(X))