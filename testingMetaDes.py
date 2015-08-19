__author__ = 'Martin'
import Helpers
import numpy as np
# from ELMimplementacije.PythonELM.elm import GenELMClassifier
# from ELMimplementacije.PythonELM.random_layer import RandomLayer
from ELMImplementacije.PythonELM.elm import GenELMClassifier
from ELMImplementacije.PythonELM.random_layer import RandomLayer
import os
from MetaDES.MetaDES import MetaDES
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from MetaDES.HelpersMeta import dviganjeDecilov
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sys import getsizeof

from sklearn.ensemble import RandomForestClassifier

# from SoftMaxRegression.softMaxSKLearn import softMaxSklearn

def main1():
    #method for testing MetaDES
    X, Y = Helpers.readData()
    divideDataForMeta(X,Y)

def divideDataForMeta(X,Y):
    #method divides data in Production, meta and selection data pieces for algorithm to run properly
    #implementirano na zacetku, najbrz ne potrebujemo te metode
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
    #Prebere potrebne podatke
    print("We are reading ")
    XProd = np.loadtxt(folder + "XProduction.csv", delimiter=",")
    YProd = np.loadtxt(folder + "YProduction.csv", delimiter=",")
    XMeta = np.loadtxt(folder + "XMeta.csv", delimiter=",")
    YMeta = np.loadtxt(folder + "YMeta.csv", delimiter=",")
    XSel = np.loadtxt(folder + "XSel.csv", delimiter=",")
    YSel = np.loadtxt(folder + "YSel.csv", delimiter=",")

    return XProd, YProd, XMeta, YMeta, XSel, YSel

def overproduction(XProd,YProd, XMeta, XSel, XTest):
    #metoda producira nove klasifikatorje za problem
    #metoda izdelana za testiranje metaDes, ne potrebujemo, ker pridobimo klasifikatorje od ostalih
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
    #metoda izdelana za testiranje metaDes, ne potrebujemo, ker pridobimo klasifikatorje od ostalih
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
def overproductionRf(XProd,YProd, XMeta, XSel, XTest, nrOfCls = 5, folder = "data/dataForMeta/"):
    #produces lot of cls for problem
    #metoda izdelana za testiranje metaDes, ne potrebujemo, ker pridobimo klasifikatorje od ostalih
    for i in range(nrOfCls):
        cls = RandomForestClassifier(n_estimators=np.random.randint(1000,1200), n_jobs=8)
        cls.name1 = "rf"+str(i)+"_"+str(cls.n_estimators)
        print("Producing " + cls.name1)
        YCaProduction, YCaMeta, YCaSel, YCaTest = trainClsForMeta(XProd, YProd, XMeta, XSel, XTest, cls)
        #save in file
        if(not os.path.isdir(folder + "classifiers/"+cls.name1)): os.makedirs(folder + "classifiers/"+cls.name1)
        # np.savetxt(folder + "classifiers/"+cls.name1+"/YCaProd.csv", YCaProduction, delimiter="\n")
        np.savetxt(folder + "classifiers/"+cls.name1+"/YCaMeta.csv", YCaMeta, delimiter="\n")
        np.savetxt(folder + "classifiers/"+cls.name1+"/YCaSel.csv", YCaSel, delimiter="\n")
        np.savetxt(folder + "classifiers/"+cls.name1+"/YCaTest.csv", YCaTest, delimiter="\n")

def overproductionELM(XProd,YProd, XMeta, XSel, XTest, nrOfCls = 5, folder = "data/dataForMeta/"):
    #produces lot of cls for problem
    #metoda izdelana za testiranje metaDes, ne potrebujemo, ker pridobimo klasifikatorje od ostalih
    for i in range(nrOfCls):
        cls = GenELMClassifier(hidden_layer = RandomLayer(n_hidden = np.random.randint(20,500), activation_func = 'multiquadric', alpha=1))
        cls.name1 = "elm_"+str(cls.hidden_layer.n_hidden)
        print("Producing " + cls.name1)
        YCaProduction, YCaMeta, YCaSel, YCaTest = trainClsForMeta(XProd, YProd, XMeta, XSel, XTest, cls)
        #save in file
        if(not os.path.isdir(folder + "classifiers/"+cls.name1)): os.makedirs(folder + "classifiers/"+cls.name1)
        # np.savetxt(folder + "classifiers/"+cls.name1+"/YCaProd.csv", YCaProduction, delimiter="\n")
        np.savetxt(folder + "classifiers/"+cls.name1+"/YCaMeta.csv", YCaMeta, delimiter="\n")
        np.savetxt(folder + "classifiers/"+cls.name1+"/YCaSel.csv", YCaSel, delimiter="\n")
        np.savetxt(folder + "classifiers/"+cls.name1+"/YCaTest.csv", YCaTest, delimiter="\n")
def overproductionProcess(folder):
    #metoda izdelana za testiranje metaDes, ne potrebujemo, ker pridobimo klasifikatorje od ostalih
    XProd, YProd = Helpers.readData()
    # divideDataForMeta(X, Y) #it divides data into Production, Meta and Selection
    XMeta, YMeta, XSel, YSel, XTest, YTest = readForMeta2(folder)
    overproductionELM(XProd,YProd, XMeta, XSel, XTest, folder=folder) #we generate classifiers and use them for responses
    overproductionRf(XProd,YProd, XMeta, XSel, XTest, folder=folder)
def readForMeta2(folder):
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

def wholeMetaProcedure2(folder = "data/dataForMeta/ostanek/"):
    #we modify this function a little bit, preparing it for work with OstanekTrain
    #S to metodo poganjamo na ostanku, metoda je bila zgrajena, da vidimo, kako se algoritmi obnasajo, pravo testiranje
    #je v naslednji metodi!
    # divideDataForMeta(X, Y) #it divides data into Production, Meta and Selection
    XMeta, YMeta, XSel, YSel, XTest, YTest = readForMeta2(folder = folder)
    # overproductionRf(XProd,YProd, XMeta, XSel, XTest) #we generate classifiers and use them for responses

    lr = LogisticRegression()




    YCaMeta = readClsResponse("Meta", folderOfClassifiers = folder+"classifiers/") #we read all classifications for meta dataset

    metaDes = MetaDES(1,300, 300, lr, competenceTresshold=0.5, mode="weighted", metaClsMode="combined",
                      nrOfClassifiers = YCaMeta.shape[1], normalizeMetaFeat=True)

    metaDes.fit(XMeta, YMeta, YCaMeta, folder = folder)
    # rf.name = "RandomForest"
    # metaDes.nrOfClassifiers = 11
    # metaDes.fitWithAlreadySaved(saveModel = False, folder = folder) #if we already computed features
    # metaDes.loadMetaCls()

    YCaSel = readClsResponse("Sel", folderOfClassifiers = folder+"classifiers/")
    YCaTest = readClsResponse("Test", folderOfClassifiers = folder+"classifiers/")
    responseTest = metaDes.predict_proba(XTest, YCaTest, XSel, YSel, YCaSel)
    name = "hC"+str(metaDes.hC)+\
                                       "_K"+str(metaDes.K)+\
                                       "_Kp"+str(metaDes.Kp)+\
                                       "_mode"+metaDes.mode+\
                                       "_competence"+str(metaDes.competenceTresshold)+\
                                       "_cls"+metaDes.metaCls.name+\
                                        "_metric"+metaDes.metric+\
                                        "_metaClsMode"+metaDes.metaClsMode
    np.savetxt(folder + "MetaDesResponse_"+name+".csv",responseTest, delimiter=",")
    responseTest = responseTest[:,1]

    plotClassifiers(folder, "MetaDesResponse_"+metaDes.mode+".csv")

def wholeMetaProcedureBackTest(folder = "data/dataForMeta/ostanek/"):
    #we modify this function a little bit, preparing it for work with backtesting
    XMeta, YMeta, XSel, YSel = readForMeta3(folder = folder)
    regionSize = XMeta.shape[0]

    cls = LogisticRegression()
    cls.name = "lr"


    hc = 1.0 #degree of consensus
    Kp = K = 52 #number of neighbors

    clsList = os.listdir(folder+"backtest/classifiers/") #["SVM1", "SVM2", "RK", "IGR_Entropy", "UniformEntropy"]
    print(clsList)

    #we read all classifications for meta dataset
    YCaMeta = readClsResponseDeterminedCls("Meta", folderOfClassifiers=folder+"classifiers/", clsList=clsList)


    metaDes = MetaDES(hc,K, Kp, cls, competenceTresshold=0.5, mode="weighted", metaClsMode="combined",
                      nrOfClassifiers = YCaMeta.shape[1], normalizeMetaFeat=True)

    name = "regionSize"+str(regionSize) + "_hC"+str(metaDes.hC)+\
                                       "_K"+str(metaDes.K)+\
                                       "_Kp"+str(metaDes.Kp)+\
                                       "_mode"+metaDes.mode+\
                                       "_competence"+str(metaDes.competenceTresshold)+\
                                       "_cls"+metaDes.metaCls.name+\
                                        "_metric"+metaDes.metric+\
                                        "_metaClsMode"+metaDes.metaClsMode+\
                                        "_clsUsed"+"_".join(clsList)
    print(name)
    metaDes.fit(XMeta, YMeta, YCaMeta, folder = folder)
    # rf.name = "RandomForest"
    # metaDes.nrOfClassifiers = 11
    # metaDes.fitWithAlreadySaved(saveModel = False, folder = folder) #if we already computed features
    # metaDes.loadMetaCls()


    YCaSel = readClsResponseDeterminedCls("Sel", folderOfClassifiers = folder+"classifiers/", clsList=clsList)
    YCaTest = readClsResponseDeterminedCls("Test", folderOfClassifiers = folder+"backtest/classifiers/", clsList=clsList)

    writeModelPreds(metaDes,folder + "MetaDesResponseFulltest_"+name+".csv", folder + "fullTestFeaturesBrezCudni.csv",
                    XSel, YSel, YCaSel, YCaTest, stp=10000, hasTicker=False)

    # plotClassifiers(folder, "MetaDesResponse_"+metaDes.mode+".csv")
def plotClassifiers(folder = "data/dataForMeta/", clsResponse = "MetaDesResponse.csv"):
    #plots all classifiers, that are stored in data/dataForMeta/classifiers, so we can compare precision with respect to
    #meta classifier
    handles = []
    YMetaDes = np.loadtxt(folder+clsResponse, delimiter=",")[:,1]
    YTest = np.loadtxt(folder + "Ytest.csv", delimiter="\n")
    YCaTest = readClsResponse("Test", folderOfClassifiers = folder+"classifiers/")
    handles.append(dviganjeDecilov(YTest, YMetaDes, "MetaDes", linewidth=4)[1])
    clsNames = os.listdir(folder+"classifiers/")
    for i, YCa in enumerate(YCaTest.T):
        print(clsNames[i],)
        res, handle = dviganjeDecilov(YTest, YCa, clsNames[i])
        handles.append(handle)
        plt.text(0.95,res[5],clsNames[i])
    plt.legend(handles = handles, loc = 2)

    # plt.savefig(folder+"krneki.png", bbox_inches='tight')
    plt.show()


def readClsResponse(datasetMode, folderOfClassifiers = "data/dataForMeta/classifiers"): #dataset = "Meta" or "Sel" or "Prod" or "Test"
    #We read all clsResponses that are in folderOfClassifiers, possible modes are "Meta", "Sel", "Test"
    YCa = []
    for cls in os.listdir(folderOfClassifiers):
        YCa.append(list(np.loadtxt(folderOfClassifiers+cls+"/YCa"+datasetMode+".csv",delimiter="\n")))
        print(folderOfClassifiers+cls)
    YCa = np.array(YCa).T
    return YCa
def readClsResponseDeterminedCls(datasetMode,clsList, folderOfClassifiers = "data/dataForMeta/classifiers/"): #dataset = "Meta" or "Sel" or "Prod" or "Test"
    #We read some, determined by clsList, clsResponses that are in folderOfClassifiers, possible modes are "Meta", "Sel", "Test"

    YCa = []
    for cls in os.listdir(folderOfClassifiers):
        if(cls in clsList):
            YCa.append(list(np.loadtxt(folderOfClassifiers+cls+"/YCa"+datasetMode+".csv",delimiter="\n")))
            print(folderOfClassifiers+cls)
    YCa = np.array(YCa).T
    return YCa
def trainClsForMeta(XProduction, YProduction,XMeta, XSel, XTest, cls):
    #method written for overproduction, we do not need it, if we already have classifiers

    cls.fit(XProduction,YProduction)
    YCaProduction = cls.predict_proba(XProduction)[:,1]
    YCaMeta = cls.predict_proba(XMeta)[:,1]
    YCaSel = cls.predict_proba(XSel)[:,1]
    YCaTest = cls.predict_proba(XTest)[:,1]

    return YCaProduction, YCaMeta, YCaSel, YCaTest

def writeModelPreds(cls, modelToWrite, dataCsvXTest,XSel, YSel, YCaSel, YCaTest,
                    stp = 10000, hasTicker = True, appendTicker = False):
    #za backtest
    #Predict fullTest
    #method for increment writting of huge datasets
    # cls = joblib.load(modelFile)
    # rnn = joblib.load("transformers/RemoveNonRanked/RemoveNonRanked.p")
    test_x = []
    tickers = []
    lines = 0
    with open(modelToWrite,"w") as fw:
        with open(dataCsvXTest,"r") as f:
            f.readline()#dummy read for header

            i=0
            for line in f:
                lines+=1
                l = line.replace("\"", "").strip().split(",")[2:] if hasTicker else line.replace("\"", "").strip().split(",")
                if hasTicker: tickers.append(line.replace("\"", "").strip().split(",")[:2])
                test_x.append(l)#izbrisemo ticker in datum
                if(len(test_x)>=stp):
                    print (i,976800/stp)
                    i+=1
                    XTest = np.array(test_x).astype(float)
                    YCaTestCut = YCaTest[(i-1)*stp:(i)*stp]
                    print("sizes are %d and %d" %(len(XTest), len(YCaTestCut)))
                    responseTest = cls.predict_proba(XTest, YCaTestCut, XSel, YSel, YCaSel)

                    preds = responseTest[:,1]
                    if hasTicker and appendTicker:
                        [fw.write("%s,%s,%s\n" %(tick[0], tick[1], str(p))) for p, tick in zip(preds, tickers)]
                    else:
                        [fw.write("%s\n" %str(p)) for p in preds]
                    test_x = []
                    tickers = []
        if(len(test_x) != 0):
            XTest = np.array(test_x).astype(float)

            responseTest = cls.predict_proba(XTest, YCaTest[(i-1)*stp:(i)*stp], XSel, YSel, YCaSel)

            preds = responseTest[:,1]
            if hasTicker and appendTicker:
                [fw.write("%s,%s,%s\n" %(tick[0], tick[1], str(p))) for p, tick in zip(preds, tickers)]
            else:
                [fw.write("%s\n" %str(p)) for p in preds]
    print("we had %d lines" %lines)

if __name__ == "__main__":
    folder = "data/dataForMeta/ostanek/"
    # overproductionProcess(folder)
    wholeMetaProcedureBackTest(folder)
    # wholeMetaProcedure2(folder)
    # plotClassifiers(folder = folder, clsResponse="MetaDesResponse_weighted.csv")

