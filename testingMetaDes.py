__author__ = 'Martin'
import Helpers
import numpy as np
# from ELMimplementacije.PythonELM.elm import GenELMClassifier
# from ELMimplementacije.PythonELM.random_layer import RandomLayer
import os
from MetaDES.MetaDES import MetaDES
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from MetaDES.HelpersMeta import dviganjeDecilov
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from sklearn.ensemble import RandomForestClassifier

def main1():
    #method for testing MetaDES
    X, Y = Helpers.readData()
    divideDataForMeta(X,Y)

def divideDataForMeta(X,Y):
    #method divides data in Production, meta and selection data pieces for algorithm to run properly

    quart = int(len(X)/4)
    XProduction, YProduction = X[:2*quart], Y[:2*quart]
    XMeta, YMeta = X[2*quart:3*quart], Y[2*quart:3*quart]
    XSel, YSel = X[3*quart:], Y[3*quart:]

    np.savetxt("data/dataForMeta/XProduction.csv", XProduction, delimiter = ",")
    np.savetxt("data/dataForMeta/YProduction.csv", YProduction, delimiter = ",")
    np.savetxt("data/dataForMeta/XMeta.csv", XMeta, delimiter = ",")
    np.savetxt("data/dataForMeta/YMeta.csv", YMeta, delimiter = ",")
    np.savetxt("data/dataForMeta/XSel.csv", XSel, delimiter = ",")
    np.savetxt("data/dataForMeta/YSel.csv", YSel, delimiter = ",")

def readForMeta(folder = "data/dataForMeta/"):
    print("We are reading ")
    XProd = np.loadtxt(folder + "XProduction.csv", delimiter=",")
    YProd = np.loadtxt(folder + "YProduction.csv", delimiter=",")
    XMeta = np.loadtxt(folder + "XMeta.csv", delimiter=",")
    YMeta = np.loadtxt(folder + "YMeta.csv", delimiter=",")
    XSel = np.loadtxt(folder + "XSel.csv", delimiter=",")
    YSel = np.loadtxt(folder + "YSel.csv", delimiter=",")

    return XProd, YProd, XMeta, YMeta, XSel, YSel

def overproduction(XProd,YProd, XMeta, XSel, XTest):
    elmc1 = GenELMClassifier(hidden_layer = RandomLayer(n_hidden = 20, activation_func = 'multiquadric', alpha=0.8))
    elmc1.name1 = "elmc1"
    elmc2 = GenELMClassifier(hidden_layer = RandomLayer(n_hidden = 50, activation_func = 'multiquadric', alpha=0.8))
    elmc2.name1 = "elmc2"
    classifiers = [elmc1, elmc2]
    for cls in classifiers:
        print("Training cls: "+cls.name1)
        YCaProduction, YCaMeta, YCaSel, YCaTest = trainClsForMeta(XProd, YProd, XMeta, XSel, XTest, cls)
        #save in file
        if(not os.path.isdir("data/dataForMeta/classifiers/"+cls.name1)): os.makedirs("data/dataForMeta/classifiers/"+cls.name1)
        np.savetxt("data/dataForMeta/classifiers/"+cls.name1+"/YCaProd.csv", YCaProduction, delimiter="\n")
        np.savetxt("data/dataForMeta/classifiers/"+cls.name1+"/YCaMeta.csv", YCaMeta, delimiter="\n")
        np.savetxt("data/dataForMeta/classifiers/"+cls.name1+"/YCaSel.csv", YCaSel, delimiter="\n")
        np.savetxt("data/dataForMeta/classifiers/"+cls.name1+"/YCaTest.csv", YCaTest, delimiter="\n")
def overproduction2(XProd,YProd, XMeta, XSel, XTest, nrOfCls = 20):
    #produces lot of cls for problem
    for i in range(nrOfCls):
        print("Producing elm " + str(i))
        cls = GenELMClassifier(hidden_layer = RandomLayer(n_hidden = np.random.randint(300,400),
                                                    activation_func = 'multiquadric', alpha=np.random.random()/2+0.5))
        cls.name1 = "elmcHigh"+str(i)
        YCaProduction, YCaMeta, YCaSel, YCaTest = trainClsForMeta(XProd, YProd, XMeta, XSel, XTest, cls)
        #save in file
        if(not os.path.isdir("data/dataForMeta/classifiers/"+cls.name1)): os.makedirs("data/dataForMeta/classifiers/"+cls.name1)
        np.savetxt("data/dataForMeta/classifiers/"+cls.name1+"/YCaProd.csv", YCaProduction, delimiter="\n")
        np.savetxt("data/dataForMeta/classifiers/"+cls.name1+"/YCaMeta.csv", YCaMeta, delimiter="\n")
        np.savetxt("data/dataForMeta/classifiers/"+cls.name1+"/YCaSel.csv", YCaSel, delimiter="\n")
        np.savetxt("data/dataForMeta/classifiers/"+cls.name1+"/YCaTest.csv", YCaTest, delimiter="\n")
def overproduction3(XProd,YProd, XMeta, XSel, XTest, nrOfCls = 10):
    #produces lot of cls for problem
    for i in range(nrOfCls):
        print("Producing rf " + str(i))
        cls = RandomForestClassifier(n_estimators=np.random.randint(80,120))
        cls.name1 = "rf"+str(i)+"_"+str(cls.n_estimators)
        YCaProduction, YCaMeta, YCaSel, YCaTest = trainClsForMeta(XProd, YProd, XMeta, XSel, XTest, cls)
        #save in file
        if(not os.path.isdir("data/dataForMeta/classifiers/"+cls.name1)): os.makedirs("data/dataForMeta/classifiers/"+cls.name1)
        np.savetxt("data/dataForMeta/classifiers/"+cls.name1+"/YCaProd.csv", YCaProduction, delimiter="\n")
        np.savetxt("data/dataForMeta/classifiers/"+cls.name1+"/YCaMeta.csv", YCaMeta, delimiter="\n")
        np.savetxt("data/dataForMeta/classifiers/"+cls.name1+"/YCaSel.csv", YCaSel, delimiter="\n")
        np.savetxt("data/dataForMeta/classifiers/"+cls.name1+"/YCaTest.csv", YCaTest, delimiter="\n")

def overproduction4(XProd,YProd, XMeta, XSel, XTest, nrOfCls = 100):
    #produces lot of cls for problem
    for i in range(nrOfCls):
        cls = RandomForestClassifier(n_estimators=np.random.randint(20,100))
        cls.name1 = "rf"+str(i)
        YCaProduction, YCaMeta, YCaSel, YCaTest = trainClsForMeta(XProd, YProd, XMeta, XSel, XTest, cls)
        #save in file
        if(not os.path.isdir("data/dataForMeta/classifiers/"+cls.name1)): os.makedirs("data/dataForMeta/classifiers/"+cls.name1)
        np.savetxt("data/dataForMeta/classifiers/"+cls.name1+"/YCaProd.csv", YCaProduction, delimiter="\n")
        np.savetxt("data/dataForMeta/classifiers/"+cls.name1+"/YCaMeta.csv", YCaMeta, delimiter="\n")
        np.savetxt("data/dataForMeta/classifiers/"+cls.name1+"/YCaSel.csv", YCaSel, delimiter="\n")
        np.savetxt("data/dataForMeta/classifiers/"+cls.name1+"/YCaTest.csv", YCaTest, delimiter="\n")
def overproductionProcess():
    X, Y = Helpers.readData()
    t = int(len(X)*0.9)
    X, Y, XTest, YTest = X[:t], Y[:t], X[t:], Y[t:]
    # divideDataForMeta(X, Y) #it divides data into Production, Meta and Selection
    XProd, YProd, XMeta, YMeta, XSel, YSel = readForMeta()
    overproduction2(XProd,YProd, XMeta, XSel, XTest) #we generate classifiers and use them for responses
def readForMeta2(folder = "data/dataForMeta/"):
    #reads for meta, when we already have overproduction
    print("we are reading in folder %s" %folder)
    XMeta = np.loadtxt(folder + "XMeta.csv", delimiter=",")
    YMeta = np.loadtxt(folder + "YMeta.csv", delimiter=",")
    XSel = np.loadtxt(folder + "XSel.csv", delimiter=",")
    YSel = np.loadtxt(folder + "YSel.csv", delimiter=",")
    XTest = np.loadtxt(folder + "XTest.csv", delimiter=",")
    YTest = np.loadtxt(folder + "YTest.csv", delimiter=",")
    print("done")
    return XMeta, YMeta, XSel, YSel, XTest,YTest
def readForMeta3(folder = "data/dataForMeta/"):
    #reads only Meta and Sel data
    print("we are reading in folder %s" %folder)
    XMeta = np.loadtxt(folder + "XMeta.csv", delimiter=",")
    YMeta = np.loadtxt(folder + "YMeta.csv", delimiter=",")
    XSel = np.loadtxt(folder + "XSel.csv", delimiter=",")
    YSel = np.loadtxt(folder + "YSel.csv", delimiter=",")
    print("done")

    return XMeta, YMeta, XSel, YSel

def wholeMetaProcedure(folder = "data/dataForMeta/"):
    X, Y = Helpers.readData()
    t = int(len(X)*0.9)
    X, Y, XTest, YTest = X[:t], Y[:t], X[t:], Y[t:]
    # np.savetxt("data/dataForMeta/Xtest.csv", XTest, delimiter=",")
    # np.savetxt("data/dataForMeta/Ytest.csv", YTest, delimiter="\n")
    # divideDataForMeta(X, Y) #it divides data into Production, Meta and Selection
    XProd, YProd, XMeta, YMeta, XSel, YSel = readForMeta()
    # overproduction3(XProd,YProd, XMeta, XSel, XTest) #we generate classifiers and use them for responses
    nb = GaussianNB()#meta classifier for metaDes
    rf = RandomForestClassifier(n_estimators=100)
    elm = GenELMClassifier(hidden_layer = RandomLayer(n_hidden = 20, activation_func = 'multiquadric', alpha=1))
    lr = LogisticRegression()

    metaDes = MetaDES(1,50, 50, lr, competenceTresshold=0.5, mode="weightedAll")


    # YCaMeta = readClsResponse("Meta") #we read all classifications for meta dataset
    # metaDes.fit(XMeta, YMeta, YCaMeta)
    # rf.name = "RandomForest"
    # metaDes.fitWithAlreadySaved(saveModel = False) #if we already computed features
    # metaDes.loadMetaCls()
    # Helpers.shraniModel(metaDes,folder+"models/metaDes.p")
    metaDes = joblib.load(folder+"models/metaDes/metaDes.p")

    YCaSel = readClsResponse("Sel")
    YCaTest = readClsResponse("Test")
    responseTest = metaDes.predict_proba(XTest, YCaTest, XSel, YSel, YCaSel)
    np.savetxt(folder + "MetaDesResponse.csv",responseTest, delimiter=",")
    responseTest = responseTest[:,1]

    handles = []

    handles.append(dviganjeDecilov(YTest, responseTest, "metaDes")[1])
    plt.legend(handles = handles, loc = 2)

    plt.show()

def wholeMetaProcedure2():
    #we modify this function a little bit, preparing it for work with OstanekTrain
    folder = "data/dataForMeta/ostanek/"
    # divideDataForMeta(X, Y) #it divides data into Production, Meta and Selection
    XMeta, YMeta, XSel, YSel, XTest, YTest = readForMeta2(folder = folder)
    # overproduction3(XProd,YProd, XMeta, XSel, XTest) #we generate classifiers and use them for responses
    nb = GaussianNB()#meta classifier for metaDes
    rf = RandomForestClassifier(n_estimators=100)
    elm = GenELMClassifier(hidden_layer = RandomLayer(n_hidden = 20, activation_func = 'multiquadric', alpha=1))
    lr = LogisticRegression()

    metaDes = MetaDES(0.8,1000, 50, lr, competenceTresshold=0.5, mode="weightedAll")


    YCaMeta = readClsResponse("Meta", folder=folder) #we read all classifications for meta dataset
    metaDes.fit(XMeta, YMeta, YCaMeta, folder = folder)
    # rf.name = "RandomForest"
    # metaDes.fitWithAlreadySaved(saveModel = False, folder = folder) #if we already computed features
    # metaDes.loadMetaCls()

    YCaSel = readClsResponse("Sel", folder = folder)
    YCaTest = readClsResponse("Test", folder = folder)
    responseTest = metaDes.predict_proba(XTest, YCaTest, XSel, YSel, YCaSel)
    np.savetxt(folder + "MetaDesResponse_"+metaDes.mode+".csv",responseTest, delimiter=",")
    responseTest = responseTest[:,1]

    handles = []

    handles.append(dviganjeDecilov(YTest, responseTest, "metaDes")[1])
    plt.legend(handles = handles, loc = 2)

    plt.show()
def plotClassifiers(folder = "data/dataForMeta/", clsResponse = "MetaDesResponse.csv"):
    #plots all classifiers, that are stored in data/dataForMeta/classifiers, so we can compare precision with respect to
    #meta classifier
    handles = []
    YMetaDes = np.loadtxt(folder+clsResponse, delimiter=",")#[:,1]
    YTest = np.loadtxt(folder + "Ytest.csv", delimiter="\n")
    YCaTest = readClsResponse("Test", folder = folder)
    handles.append(dviganjeDecilov(YTest, YMetaDes, "MetaDes", linewidth=4)[1])
    clsNames = os.listdir(folder+"classifiers/")
    for i, YCa in enumerate(YCaTest.T):
        print(clsNames[i],)
        handles.append(dviganjeDecilov(YTest, YCa, clsNames[i])[1])
    plt.legend(handles = handles, loc = 2)

    # plt.savefig(folder+"krneki.png", bbox_inches='tight')
    plt.show()


def readClsResponse(datasetMode, folder = "data/dataForMeta/"): #dataset = "Meta" or "Sel" or "Prod" or "Test"
    YCa = []
    for cls in os.listdir(folder+"classifiers/"):
        YCa.append(list(np.loadtxt(folder+"classifiers/"+cls+"/YCa"+datasetMode+".csv",delimiter="\n")))
        print(folder+"classifiers/"+cls)
    YCa = np.array(YCa).T
    return YCa
def trainClsForMeta(XProduction, YProduction,XMeta, XSel, XTest, cls):

    cls.fit(XProduction,YProduction)
    YCaProduction = cls.predict_proba(XProduction)[:,1]
    YCaMeta = cls.predict_proba(XMeta)[:,1]
    YCaSel = cls.predict_proba(XSel)[:,1]
    YCaTest = cls.predict_proba(XTest)[:,1]

    return YCaProduction, YCaMeta, YCaSel, YCaTest

if __name__ == "__main__":
    folder = "data/dataForMeta/ostanek/"
    # overproductionProcess()
    # wholeMetaProcedure2()
    plotClassifiers(folder = "data/dataForMeta/ostanek/", clsResponse="MetaDesResponse_weighted.csv")

