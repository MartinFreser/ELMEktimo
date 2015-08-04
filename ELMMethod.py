__author__ = 'Martin'
from ELMimplementacije.PythonELM.elm import GenELMClassifier
from ELMimplementacije.PythonELM.random_layer import RandomLayer
import numpy as np
import Helpers
import pickle
from sklearn.ensemble import BaggingClassifier
from sklearn.cross_validation import train_test_split

"""
    V tej metodi iscemo optimalne parametre za metodo ELM
"""

class ELMMethod():
    def poisciParametre(self, X,Y):

        activation_functions = ['multiquadric' ]# 'softlim', 'multiquadric', 'inv_multiquadric', 'gaussian', 'tanh', 'sine', 'tribas', 'inv_tribas', 'sigmoid']

        n_hiddens = [200,300,400,500]#3, 30,50,100]
        parameters = []
        alphas = [1.0,0.7]#0.0,0.2,0.4,0.5,0.7,0.9,1.0]
        nrOfTrials = len(activation_functions)*len(alphas) * len(n_hiddens)
        trial = 1
        np.random.seed(np.random.randint(10000000))
        for n_hidden in  n_hiddens:
            for alpha in alphas:
                for actFunction in activation_functions:
                    cls = GenELMClassifier(hidden_layer = RandomLayer(n_hidden = n_hidden, activation_func = actFunction, alpha=alpha))

                    parameter = Helpers.cv(X,Y,cls,5, printing = False)
                    parameter = parameter+ [n_hidden, alpha, actFunction, "normal"]
                    parameters.append(parameter)
                    print(parameter, "%d/%d" %(trial,nrOfTrials))

                    # parameter = Helpers.cv(X,Y,BaggingClassifier(cls,n_estimators=30),10, printing = False)
                    # parameter = parameter+ [n_hidden, alpha, actFunction, "bagged"]
                    # parameters.append(parameter)
                    # print(parameter, "%d/%d" %(trial,nrOfTrials))

                    trial = trial+1
        pickle.dump(parameters,open("parametersMultiQuadric.p","wb"))
        return
    def vrniMethod(self, parameter):
        # acc,prec,precTress, n_hidden, rhl, actFunction = parameter
        cls = GenELMClassifier(hidden_layer = RandomLayer(n_hidden = parameter[-3], activation_func = parameter[-1], alpha=parameter[-2]))
        return cls
        # tr, ts, trRaw, tsRaw, prec, precTress = res_dist(X,Y,cls,10)
