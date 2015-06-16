__author__ = 'Martin'
from ELMimplementacije.PythonELM.elm import ELMClassifier, GenELMClassifier, ELMRegressor, GenELMRegressor
from ELMimplementacije.PythonELM.random_layer import RandomLayer, RBFRandomLayer, GRBFRandomLayer, MLPRandomLayer
import numpy as np
import Helpers
import pickle
from sklearn.ensemble import BaggingClassifier
from sklearn.cross_validation import train_test_split
from bagging import Bagging
from sklearn.externals import joblib
from sklearn import cross_validation
from sklearn.metrics import precision_score
class BaggingMethod():
    def __init__(self,estimator):
        self.estimator = estimator
    def poisciParametre(self, X,Y):
        parameterFile = "BaggingParametersCV.p"
        # Helpers.deleteFile(parameterFile)

        parameters = []
        n_folds = 4
        for n_est in [1,20,40,60,80,100,120]:
            k_fold = cross_validation.KFold(n=X.shape[0], n_folds=n_folds, shuffle=True)

            ratioCuts = [0,0.4,0.6,0.8,0.9,0.95, 1-2.0/n_est]
            resu = np.zeros(len(ratioCuts))
            for trainIdx, testIdx in k_fold:
                baggedEst = Bagging(self.estimator, n_estimators=n_est)
                baggedEst.fit(X[trainIdx],Y[trainIdx])
                i=0
                for ratioCut in ratioCuts: #n*(1-x) = 2, we select best 2 algos
                    # printSome(baggedEst,X[testIdx])
                    baggedEst.ratioCutOffEstimators = ratioCut
                    preds = baggedEst.predict_proba(X[testIdx])
                    res = Helpers.metricWithRawDataAboveDecil(preds[:,1],Y[testIdx],0.9, precision_score)
                    resu[i]+=res
                    i+=1
            resu /= n_folds
            for i, r in enumerate(resu):
                parameter = (r,n_est,ratioCuts[i])
                print(np.array(parameter)) #we put in array just for pretty print
                parameters.append(parameter)
            # Helpers.pickleListAppend(parameters,parameterFile)
    def vrniMethod(self, parameter):
        # acc,prec,precTress, n_hidden, rhl, actFunction = parameter
        pass
def printSome(baggedEst,x):
    ests = [est for err, est in sorted(baggedEst.oobErrorsAndEstimators, reverse=True, key=lambda pair: pair[0])]
    preds = np.array([e.predict_proba(x[:50])[:,1] for e in ests]).T
    truePreds = baggedEst.predict_proba(x[:50])[:,1]
    bla = np.column_stack([preds, truePreds])
    print(bla)
    bla = 0
