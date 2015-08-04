__author__ = 'Martin'
"""
    Pomozne metode, ki so v pomoc v drugih programih. Implementirane so na primer precno preverjanje, branje podatkov,
    delo za pisanje v datoteke, ...
"""
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt
import sklearn.metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from time import time
import numpy as np
import pandas
from sklearn import cross_validation
import pickle
import os.path
from sklearn.externals import joblib
import random

np.set_printoptions(precision=3)


def cv(X, Y, cls, folds =3, random_seed = 1234, printing = False):
    #method returns train_acc, test_acc, train_acc_lastDecil, test_acc_lastDecil, precisions, precisions_lastDecil, time
    np.random.seed(random_seed)

    test_acc = []
    train_acc = []
    train_acc_lastDecil = []
    test_acc_lastDecil = []
    precisions = []
    precisions_lastDecil = []
    tresshold = 0.5
    start_time = time()

    k_fold = cross_validation.KFold(n=X.shape[0], n_folds=folds)
    for trainIdx, testIdx in k_fold:
        x_train = X[trainIdx]
        y_train = Y[trainIdx]
        x_test = X[testIdx]
        y_test = Y[testIdx]
        cls.fit(x_train, y_train)
        classPredTrain = cls.predict(x_train)
        probPredTrain = cls.predict_proba(x_train)[:,1]
        classPredTest = cls.predict(x_test)
        probPredTest = cls.predict_proba(x_test)[:,1]
        # print("y_train: # of pos: %d, #of neg:%d" %(len(y_train[y_train>0.5]), len(y_train[y_train<0.5])))
        # print("classPredTrain: # of pos: %d, #of neg:%d" %(len(classPredTrain[classPredTrain>0.5]), len(classPredTrain[classPredTrain<0.5])))
        # print("classPredTest: # of pos: %d, #of neg:%d" %(len(classPredTest[classPredTest>0.5]), len(classPredTest[classPredTest<0.5])))
        if(printing): print("Max: %f.2  Min: %f.2 , # x>%.2f: %d, #x<-%.2f: %d"%
              (np.max(probPredTest), np.min(probPredTest),tresshold,
               len(probPredTest[probPredTest>tresshold]),tresshold, len(probPredTest[probPredTest<-tresshold])))


        train_acc.append(accuracy_score(y_train, classPredTrain))
        test_acc.append(accuracy_score(y_test, classPredTest))

        train_acc_lastDecil.append(metricWithRawData2(classPredTrain, probPredTrain,y_train, 0.1, accuracy_score))
        test_acc_lastDecil.append(metricWithRawData2(classPredTest, probPredTest,y_test, 0.1, accuracy_score))
        # if (i%(n_runs/5) == 0): print("%d"%i),

        precisions_lastDecil.append(metricWithRawData2(classPredTest, probPredTest,y_test, 0.1, precision_score))
        precisions.append(precision_score(y_test,classPredTest))


    if(printing):
        print("\nTime: %.3f secs" % (time() - start_time))

        print("Test Min: %.3f Mean: %.3f Max: %.3f SD: %.3f" % (min(test_acc), np.mean(test_acc), max(test_acc), np.std(test_acc)))
        print("Train Min: %.3f Mean: %.3f Max: %.3f SD: %.3f" % (min(train_acc), np.mean(train_acc), max(train_acc), np.std(train_acc)))
        print()
        print("With raw predicts and tresshold used: ")
        print("Test Min: %.3f Mean: %.3f Max: %.3f SD: %.3f" % (min(test_acc_lastDecil), np.mean(test_acc_lastDecil), max(test_acc_lastDecil), np.std(test_acc_lastDecil)))
        print("Train Min: %.3f Mean: %.3f Max: %.3f SD: %.3f" % (min(train_acc_lastDecil), np.mean(train_acc_lastDecil), max(train_acc_lastDecil), np.std(train_acc_lastDecil)))
    metrics = list(map(np.mean,[train_acc, test_acc, train_acc_lastDecil, test_acc_lastDecil, precisions, precisions_lastDecil, time()-start_time]))
    # return (train_acc, test_acc, train_acc_lastDecil, test_acc_lastDecil, precisions, precisions_lastDecil, time()-start_time)
    return metrics

def readData(firstNSamples=0, trainFtrFile = "data/trainFtr.csv", trainClsFile = "data/trainCls.csv",  hasTicker = False, deleteFirstNFeatures = 1):
    #load podatkov
    train_set_x = pandas.read_csv(trainFtrFile, sep=',',header=0)
    train_set_y = pandas.read_csv(trainClsFile, sep=',',header=0)
    if(firstNSamples == 0): firstNSamples = len(train_set_x.values)
    train_set_x = train_set_x.values[:firstNSamples]
    train_set_y = train_set_y.values[:firstNSamples]

    return train_set_x[:,deleteFirstNFeatures:].astype(float), train_set_y.T[0].astype(float)

def readClsResponse(file):
    res = []
    with open(file) as f:
        for line in f:
            res.append(float(line.strip()))
    return np.array(res).astype(float)

def metricWithRawData(classPredicts, rawPredicts, trueY, tresshold= 0.5, metricFunction = accuracy_score):
    if(len(classPredicts) != len(trueY)): raise Exception("Lengts of vectors classPredicts, rawPredicts, trueY should be"
                                                          "the same")
    rawPredicts = np.array(rawPredicts)
    idxs = np.where(np.logical_or(rawPredicts < -tresshold, rawPredicts>tresshold))
    metric = metricFunction(trueY[idxs],classPredicts[idxs])
    return metric

def metricWithRawData2(classPredicts, rawPredicts, trueY, decil = 0.1, metricFunction = accuracy_score):
    if(len(classPredicts) != len(trueY)): raise Exception("Lengts of vectors classPredicts, rawPredicts, trueY should be"
                                                          "the same")
    rawPredicts = np.array(rawPredicts)
    sortedPreds = sorted(rawPredicts)
    posTresshold = sortedPreds[-int(len(rawPredicts)*decil)]

    idxs = np.where(rawPredicts>=posTresshold)[0]
    metric = metricFunction(trueY[idxs],np.ones(len(idxs)))
    return metric

def metricWithRawDataAboveDecil(rawPredicts, trueY, decil = 0.1, metricFunction = accuracy_score):
    if(len(rawPredicts) != len(trueY)): raise Exception("Lengts of vectors classPredicts, rawPredicts, trueY should be"
                                                          "the same")
    rawPredicts = np.array(rawPredicts)
    sortedPreds = sorted(rawPredicts)
    posTresshold = sortedPreds[int(len(rawPredicts)*decil)]

    idxs = np.where(rawPredicts>=posTresshold)[0] #ker nam np.where vraca tuple
    metric = metricFunction(trueY[idxs],np.ones(len(idxs))) #we predict that all instances above some point will be one
    return metric

def dviganjeDecilov(x_test,y_test, fittedCls,clsName, plotResults = True, decils = [0.5,0.6,0.7,0.8,0.9,0.95]):
    # x_train, x_test, y_train, y_test = splitTrainTest(X, Y, test_size=test_size)
    results = []
    predicts = fittedCls.predict_proba(x_test)[:,1]
    for decil in decils:
        result = metricWithRawDataAboveDecil(predicts,y_test,decil, precision_score)
        results.append(result)
    #pickle.dump(results,open("dviganjeDecilovGrafi/%s.p"%clsName,"wb"))
    if (plotResults):
        handle, = plt.plot(decils,results, label = clsName)
        print(results)
    return np.array(results), handle if plotResults else None
def dviganjeDecilovKkrat(X,Y, cls,clsName, test_size = 0.1, k=3, plotResults = True):
    results = []
    for i in range(k):
        res, none = dviganjeDecilov(X,Y, cls,clsName, test_size = 0.1, plotResults=False)
        results.append(res)
    results = np.mean(np.array(results),axis = 0)
    decils = [0.5,0.6,0.7,0.8,0.9,0.95]
    if (plotResults):
        handle, = plt.plot(decils,results, label = clsName)
    print(results)
    return results, handle if plotResults else None
def splitTrainTest(x,y,test_size):
    t = int(len(y)*(1-test_size))
    return x[:t],x[t:], y[:t], y[t:]
def pickleListAppend(object, file):
    if not isinstance(object,list): assert "Object has to be list"
    if(not os.path.isfile(file)):
        pickle.dump(object,open(file,"wb"))
    else:
        obj = pickle.load(open(file,"rb"))
        pickle.dump(obj+object,open(file,"wb"))

def pickleListAppend2(object, file):
    if(not os.path.isfile(file)):
        list = [object]
        pickle.dump(list,open(file,"wb"))
    else:
        list = pickle.load(open(file,"rb"))
        list.append(object)
        pickle.dump(list,open(file,"wb"))

def deleteFile(file):
    if(os.path.exists(file)): os.remove(file)

def shraniModel(cls, file):
    if(not os.path.exists("/".join(file.split("/")[:-1]))): os.makedirs("/".join(file.split("/")[:-1]))
    joblib.dump(cls,file)
    print("shranil v ", file)

def writeModelPreds(modelFile, modelToWrite, dataCsv, stp = 10000, hasTicker = True, appendTicker = False, transformers = None):
    #za backtest
    #Predict fullTest
    cls = joblib.load(modelFile)
    # rnn = joblib.load("transformers/RemoveNonRanked/RemoveNonRanked.p")
    test_x = []
    tickers = []
    lines = 0
    with open(modelToWrite,"w") as fw:
        with open(dataCsv,"r") as f:
            f.readline()#dummy read for header

            i=0
            for line in f:
                lines+=1
                l = line.replace("\"", "").strip().split(",")[2:] if hasTicker else line.replace("\"", "").strip().split(",")
                if hasTicker: tickers.append(line.replace("\"", "").strip().split(",")[:2])
                test_x.append(l)#izbrisemo ticker in datum
                if(len(test_x)>stp):
                    print (i,976800/stp)
                    i+=1
                    test_x = np.array(test_x).astype(float) + 1 #hardcoded, need to be removed!!!
                    if transformers:
                        for t in transformers:
                            test_x = t.transform(test_x)
                    preds = cls.predict_proba(test_x)[:,1]
                    if hasTicker and appendTicker:
                        [fw.write("%s,%s,%s\n" %(tick[0], tick[1], str(p))) for p, tick in zip(preds, tickers)]
                    else:
                        [fw.write("%s\n" %str(p)) for p in preds]
                    test_x = []
                    tickers = []
        if(len(test_x) != 0):
            test_x = np.array(test_x).astype(float) + 1 #hardcoded, need to be removed
            if transformers:
                for t in transformers:
                    test_x = t.transform(test_x)
            preds = cls.predict_proba(test_x)[:,1]
            if hasTicker and appendTicker:
                [fw.write("%s,%s,%s\n" %(tick[0], tick[1], str(p))) for p, tick in zip(preds, tickers)]
            else:
                [fw.write("%s\n" %str(p)) for p in preds]
            test_x = []
            tickers = [] #tega ne bi rabili narediti
    print("we had %d lines" %lines)

def meanAndStd(x):
    mean = np.mean(x)
    std = np.sqrt((np.sum((x-mean)**2)/len(x)))

    return mean, std



