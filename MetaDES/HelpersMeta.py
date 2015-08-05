__author__ = 'Martin'
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
from matplotlib import pyplot as plt
from Helpers import metricWithRawDataAboveDecil
import Helpers
"""
    V tej datoteki implementiramo metode, ki nam pomagajo pri rezredu MetaDES()
"""
def DOC(x, mode = 0):
    #degree of consensus
    #   we have two possible modes
    if(mode == 0):
        doc = len(x[x == x[0]])/len(x)
        return max(doc,1-doc)
    elif(mode == 1):
        t1 = len(x[x == x[0]])
        t2 = len(x)-t1
        delta = abs(t1-t2)
        return delta/len(x)
# @profile
def computeMetaFeatures(reg, opReg):
    #it computes meta features as stated in article
    # we added one more feature, which computes mean and standard deviation of prediction of classifiers (f6)
    f = []
    #f1
    f1 = 1-np.abs(reg["Y"] - np.round(reg["YC"]))
    mean,std = Helpers.meanAndStd(f1)
    f.append(mean)
    f.append(std)
    #f2
    f2 = reg["YC"]
    # f = np.append(f,f2)
    mean,std = Helpers.meanAndStd(f2)
    f.append(mean)
    f.append(std)

    #f3
    # f3 = accuracy_score(reg["Y"],np.round(reg["YC"]))
    # f = np.append(f,f3)
    #f4
    # f4 = opReg["YC"]
    f4 = 1-np.abs(opReg["Y"] - np.round(opReg["YC"]))
    # f = np.append(f,f4)
    mean,std = Helpers.meanAndStd(f4)
    f.append(mean)
    f.append(std)

    #f5 to be done

    f6 = np.abs(reg["YC"]-reg["Y"])
    mean,std = Helpers.meanAndStd(f6)
    f.append(mean)
    f.append(std)

    f7 = np.abs(opReg["YC"]-opReg["Y"])
    mean,std = Helpers.meanAndStd(f7)
    f.append(mean)
    f.append(std)

    return f



def dviganjeDecilov(y_test, predicts,clsName, plotResults = True, decils = [0.5,0.6,0.7,0.8,0.9,0.95], linewidth = 1):
    # x_train, x_test, y_train, y_test = splitTrainTest(X, Y, test_size=test_size)
    results = []
    for decil in decils:
        result = metricWithRawDataAboveDecil(predicts,y_test,decil, precision_score)
        results.append(result)
    #pickle.dump(results,open("dviganjeDecilovGrafi/%s.p"%clsName,"wb"))
    if (plotResults):
        handle, = plt.plot(decils,results, label = clsName, linewidth = linewidth)
        print(results)
    return np.array(results), handle if plotResults else None