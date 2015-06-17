__author__ = 'Martin'
import pandas
from matplotlib import pyplot as plt
from ELMImplementacije.PythonELM.elm import GenELMClassifier
from ELMImplementacije.PythonELM.random_layer import RandomLayer
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



def main1():
    X, Y = readData(10000)
    elmc = ELMClassifier(n_hidden=100, activation_func='gaussian')
    baggedElmc = BaggingClassifier(elmc)

    #baggedElmc.fit(X,Y)
    rhl = RandomLayer(n_hidden=500, activation_func='gaussian')
    genElmc = GenELMClassifier(rhl)


    tr, ts, trRaw, tsRaw, prec, precTress, duration = Helpers.cv(X,Y,baggedElmc,3, printing=True)
    plt.scatter(tr, ts, alpha=0.5, marker='D', c='r')
    plt.scatter(trRaw, tsRaw, alpha=0.5, marker='D', c='b')
    plt.show()
    plt.scatter(prec, precTress,alpha=0.9, marker='D', c='r')
    plt.show()
def main2():
    X, Y = readData()

    ELMMethod.ELMMethod().poisciParametre(X,Y)
    # vrniMethod(pickle.load(open("parameter.p","rb")), X, Y)
def main2LoadResults():
    #load results produced by main2)
    file = "parameters.p" #"baggedElmcParameters.p"
    results = pickle.load(open(file,"rb"))
    results.sort(key = lambda r: r[5], reverse = True) #results.sort(key = lambda r: r[0], reverse = True)
    print ("\n".join(map(str,results)))
def main3():
    #Testira nekaj algoritmov
    X, Y = readData(trainFtrFile="data/trainFtrExtended_200f.csv",
                    trainClsFile= "data/trainClsExtended.csv", deleteFirstNFeatures=2)
    # X, Y = readData(1000)
    elmc = GenELMClassifier(hidden_layer = RandomLayer(n_hidden = 50, activation_func = 'multiquadric', alpha=1.0))
    baggedElmc = BaggingClassifier(elmc, n_estimators=10)
    ada = AdaBoostClassifier(n_estimators=30)
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators = 50, n_jobs = 1)
    np.random.seed(np.random.randint(10000000))
    adaRf = AdaBoostClassifier(rf, n_estimators=20)

    print ("elmc: ", Helpers.cv(X,Y,elmc))
    print ("baggedElmc: ", Helpers.cv(X,Y,baggedElmc))
    print ("adaTree: ", Helpers.cv(X,Y,ada))
    print ("rf: ", Helpers.cv(X,Y,rf))
    print ("adaRf: ", Helpers.cv(X,Y,adaRf))
def main4():
    #na enem algoritmu dvigujem decil na testni mnozici in gledamo precision
    # X, Y = readData()
    trainFtrFile = "//./Z:/spaceextension/test10k/csv/trainFtrExtended_200f_90.csv"
    trainClsFile = "//./Z:/spaceextension/test10k/csv/trainClsExtended_90.csv"
    X, Y = readData(trainFtrFile=trainFtrFile,
                    trainClsFile= trainClsFile,
                    deleteFirstNFeatures=2,
                    firstNSamples=1000)
    x_train, x_test, y_train, y_test = splitTrainTest(X, Y, test_size=0.1)
    print(np.mean(y_test))

    n_hidden = 100
    elmc = GenELMClassifier(hidden_layer = RandomLayer(n_hidden = n_hidden, activation_func = 'multiquadric', alpha=1))
    elmc1 = GenELMClassifier(hidden_layer = RandomLayer(n_hidden = n_hidden, activation_func = 'multiquadric', alpha=0.5))
    elmc2 = GenELMClassifier(hidden_layer = RandomLayer(n_hidden = n_hidden, activation_func = 'multiquadric', alpha=0))
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators = 1000, n_jobs = 2)
    # baggedElmc = BaggingClassifier(elmc, n_estimators=60, bootstrap= False, max_samples=0.6, max_features=0.6)
    adaTree = AdaBoostClassifier(n_estimators=30)
    baggedElmc = Bagging(elmc, n_estimators=100,ratioCutOffEstimators=0.7)
    baggedElmc2 = Bagging(elmc, n_estimators=100,ratioCutOffEstimators=0.7)
    baggedElmc3 = Bagging(elmc, n_estimators=100,ratioCutOffEstimators=0.7)
    meanEnsemble = MeanEnsemble(elmc, n_estimators=50)

    # uncertainBagging = BaggingUncertain(elmc, n_estimators=100,ratioCutOffEstimators=0.7)

    k = 3
    handles = []
    # handles.append(dviganjeDecilovKkrat(X,Y,elmc, "GenElmc",test_size=0.1,k=k)[1])

    # meanEnsemble.fit(x_train,y_train)
    # handles.append(dviganjeDecilov(x_test,y_test,meanEnsemble,"meanEnsemble ")[1])
    #
    #
    # baggedElmc.fit(x_train,y_train)
    # handles.append(dviganjeDecilov(x_test,y_test,baggedElmc,"BaggedElmc "+str(baggedElmc.n_estimators)+" "+str(baggedElmc.ratioCutOffEstimators))[1])

    elmc.fit(x_train,y_train)
    handles.append(dviganjeDecilov(x_test,y_test,elmc, "GenElmc1")[1])

    # elmc1.fit(x_train,y_train)
    # handles.append(dviganjeDecilov(x_test,y_test,elmc1, "GenElmc0.5")[1])
    #
    # elmc2.fit(x_train,y_train)
    # handles.append(dviganjeDecilov(x_test,y_test,elmc2, "GenElmc20")[1])

    rf.fit(x_train,y_train)
    handles.append(dviganjeDecilov(x_test,y_test,rf,"RandomForest")[1])

    # allRes = []
    # for n_est in [20,40,60,80]:
    #     baggedElmc = Bagging(elmc, n_estimators=n_est)
    #     baggedElmc.fit(x_train,y_train)
    #     joblib.dump(baggedElmc,"baggedElmcModels/baggedElmc_"+str(n_est)+"est.pkl")
    #     for ratioCut in [0.4,0.6,0.8,0.9,0.95]:
    #         baggedElmc.ratioCutOffEstimators = ratioCut
    #         results = dviganjeDecilov(x_test,y_test,baggedElmc,"BaggedElmc "+str(baggedElmc.n_estimators)+" "+str(ratioCut))
    #         parameter = (results[0],n_est,ratioCut)
    #         print(parameter)
    #         allRes.append(parameter)
    #         handles.append(results[1])
    # pickle.dump(allRes, open("baggedElmcParameters.p","wb"))


    # handles.append(dviganjeDecilov(X,Y,tree,"tree ",test_size=0.1)[1])
    # handles.append(dviganjeDecilov(X,Y,adaTree,"adaTree ",test_size=0.1)[1])

    adaTree.fit(x_train,y_train)
    handles.append(dviganjeDecilov(x_test,y_test,adaTree,"adaTree")[1])
    plt.legend(handles = handles, loc = 2)

    plt.show()
def main7():
    #Preverjanje standardne deviacije algoritma elm
    X, Y = readData()
    # rnn = RemoveNonRankedFeatures()
    # rnn.fit(X)
    # shraniModel(rnn, "transformers/RemoveNonRanked/RemoveNonRanked.p")
    rnn = joblib.load("transformers/RemoveNonRanked/RemoveNonRanked.p")
    X = rnn.transform(X)
    x_train, x_test, y_train, y_test = splitTrainTest(X, Y, test_size=0.1)
    results = []

    cls = GenELMClassifier(hidden_layer = RandomLayer(n_hidden = 100, activation_func = 'multiquadric', alpha=1))
    # cls = Bagging(cls, n_estimators=50,ratioCutOffEstimators=0.9)
    cls = BaggingUncertain(cls,n_estimators=50,ratioCutOffEstimators=0.9, ratioOfDeltaRemove=0.1)

    for i in range(7):
        print(i)
        cls.fit(x_train,y_train)
        results.append(dviganjeDecilov(x_test,y_test,cls, "elmc", plotResults=True, decils=[0.5,0.9])[0])
    results = np.array(results)
    print(results)
    print(results.std(axis=0))
    print(results.mean(axis = 0))

def main5():
    #Cross validation 3 algoritmov
    X, Y = readData()
    tree = DecisionTreeClassifier()
    elmc = GenELMClassifier(hidden_layer = RandomLayer(n_hidden = 100, activation_func = 'multiquadric', alpha=1.0))
    baggedelmc = Bagging(elmc, n_estimators=20,ratioCutOffEstimators=0)

    print ("baggedelmc: ", Helpers.cv(X,Y,baggedelmc))


def main9():
    #zgradi najljubse modele
    # X,Y = readData()
    # rnn = joblib.load("transformers/RemoveNonRanked/RemoveNonRanked.p")
    # X = rnn.transform(X)
    X, Y = readData(trainFtrFile="data/trainFtrExtended_200f.csv",
                    trainClsFile= "data/trainClsExtended.csv", deleteFirstNFeatures=2)
    t = len(X)*0.9
    # X, Y = X[:t], Y[:t]

    elmc = GenELMClassifier(hidden_layer = RandomLayer(n_hidden = 200, activation_func = 'multiquadric', alpha=0.9))
    elmc.fit(X,Y)
    shraniModel(elmc,"models/elm_400Extended/elm_400Extended.p")

    elmc = GenELMClassifier(hidden_layer = RandomLayer(n_hidden = 50, activation_func = 'multiquadric', alpha=0.8))
    baggedElmc = Bagging(elmc,n_estimators=50,ratioCutOffEstimators=0.5)
    baggedElmc.fit(X,Y)
    shraniModel(elmc,"models/baggedElm_50_50_0.5Extended/baggedElm_50_50_0.5Extended.p")

def main8(modelFile, writeFile):
    #PredictOstanekTest
    #napovemo verjetnosti da bodo nalozbe uspesne iz veliiiike podatkovnew zbirke
    cls = joblib.load(modelFile)
    rnn = joblib.load("transformers/RemoveNonRanked/RemoveNonRanked.p")
    stp = 4000
    with open(writeFile,"w") as fw:
        for i in range(10):
            print(i,"/",9)
            test_x = pandas.read_csv("//./Z:/Podatki/Prediction datasets/csv/testFtrAll"+str(i), sep=',',header=0)
            test_x = test_x.values[:]
            test_x = rnn.transform(test_x)
            for j in range(0,len(test_x),stp):
                preds = cls.predict_proba(test_x[j:j+stp])[:,1]
                [fw.write(str(p)+"\n") for p in preds]


def main10():
    model = "elm_400Extended"
    # csv = "//./Z:/Podatki/Prediction datasets/ostanekTrainFeaturesWithTickers.csv"
    csv = "//./Z:/spaceextension/testFtrExtended_200f.csv"
    # csv = "//./Z:/Podatki/Prediction datasets/fullTestFeatures.csv"#"data/fullTestFeaturesMaliKos.csv"
    # csv = "data/testFtrWithTicker.csv"

    toWrite = "models/"+model+"/testFtrExtended_200fPredictions.csv"
    # toWrite = "models/"+model+"/smallTestResponse.csv"
    # toWrite = "models/"+model+"/predictionsFullTest2.csv"
    # main8("models/"+model+"/"+model+".p", "models/"+model+"/predictionsOstanek.csv", )
    Helpers.writeModelPreds("models/"+model+"/"+model+".p",
           toWrite,
           csv,
           hasTicker=True,
           appendTicker=False)

def main6():
    #Cross validation iskanje parametrov za Bagging elmcje
    X,Y = readData()
    elmc = GenELMClassifier(hidden_layer = RandomLayer(n_hidden = 100, activation_func = 'multiquadric', alpha=1.0))
    BaggingMethod.BaggingMethod(elmc).poisciParametre(X,Y)
def bla():
    with open("//./Z:/krneki.txt") as f:
        for line in f:
            print(line)
if __name__ == "__main__":
    # bla()
    main4()
    # main2LoadResults()

