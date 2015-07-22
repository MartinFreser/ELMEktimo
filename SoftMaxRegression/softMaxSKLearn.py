__author__ = 'Martin'
import Orange
import numpy as np
class softMaxSklearn:
    def __init__(self, lambda_ = 1.0):
        self.lambda_ = lambda_
    def fit(self,X,Y):
        cls = Orange.classification.SoftmaxRegressionLearner(lambda_=self.lambda_)
        data = Orange.data.Table(X,Y)
        self.model = cls(data)
        self.domain = data.domain
    def predict_proba(self,X):
        X = np.array(X)
        data= Orange.data.Table(self.domain,X, np.zeros(X.shape[0]))
        return self.model(data, 1)
if __name__ == "__main__":
    X = np.random.rand(4,10)
    Y = np.array([[0],[1],[1],[0]])
    sm = softMaxSklearn()
    sm.fit(X,Y)
    print(sm.predict_proba(X))