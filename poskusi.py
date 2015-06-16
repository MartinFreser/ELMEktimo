__author__ = 'Martin'
import pandas
from matplotlib import pyplot as plt
from ELMimplementacije.PythonELM.elm import ELMClassifier, GenELMClassifier, ELMRegressor, GenELMRegressor
from ELMimplementacije.PythonELM.random_layer import RandomLayer, RBFRandomLayer, GRBFRandomLayer, MLPRandomLayer
from time import time
import numpy as np
from Helpers import readData, dviganjeDecilov, dviganjeDecilovKkrat, splitTrainTest, shraniModel
import ELMMethod, BaggingMethod
import pickle
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
import sklearn
from bagging import Bagging, BaggingUncertain
from sklearn import cross_validation
from meanEnsemble import MeanEnsemble
import Helpers
import os
from RemoveNonRankedFeatures import RemoveNonRankedFeatures

def saveLoadModels():
    #preverimo ali shranjevanje in nalaganje modelov sploh deluje
    X,Y = readData(1000)
    elmc = GenELMClassifier(hidden_layer = RandomLayer(n_hidden = 20, activation_func = 'multiquadric', alpha=1.0))
    baggedelmc = Bagging(elmc, n_estimators=20,ratioCutOffEstimators=0.5)
    baggedelmc.fit(X,Y)
    shraniModel(baggedelmc, "models/baggedElmc_20_0.5/baggedElmc.p")
    belmc = joblib.load("models/baggedElmc_20_0.5/baggedElmc.p")
    print(belmc.predict(X[:20]))