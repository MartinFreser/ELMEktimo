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

from Helpers import pickleListAppend2

from sklearn.ensemble import RandomForestClassifier
from testingMetaDes import readForMeta2, readClsResponse
from sklearn.neighbors.ball_tree import BallTree

def findParameters():
    #we modify this function a little bit, preparing it for work with OstanekTrain
    folder = "data/dataForMeta/ostanek/"
    # divideDataForMeta(X, Y) #it divides data into Production, Meta and Selection
    XMeta, YMeta, XSel, YSel, XTest, YTest = readForMeta2(folder = folder)

    nb = GaussianNB()#meta classifier for metaDes
    nb.name="Bayes"
    rf = RandomForestClassifier(n_estimators=50)
    rf.name="rf"
    # elm = GenELMClassifier(hidden_layer = RandomLayer(n_hidden = 100, activation_func = 'multiquadric', alpha=1))
    # elm.name="elm"
    lr = LogisticRegression()
    lr.name= "lr"

    metaClassifiers = [lr]
    hCs = [1.0,0.8,0.6]
    nrNeigh = [300,50,1000]
    modes = ["weightedAll", "weighted", "mean"]
    metrics = ["chebyshev", "l2"]# BallTree.valid_metrics
    competenceTressholds = [0.4,0.5,0.6]

    # metaDes = MetaDES(0.8,1000, 50, lr, competenceTresshold=0.5, mode="weightedAll")


    YCaMeta = readClsResponse("Meta", folder=folder) #we read all classifications for meta dataset
    YCaSel = readClsResponse("Sel", folder = folder)
    YCaTest = readClsResponse("Test", folder = folder)

    for nrN in nrNeigh:
        for hC in hCs:
            for mode in modes:
                for metric in metrics:
                    try:
                        metaDes = MetaDES(hC,nrN, nrN, lr, competenceTresshold=0.5, mode=mode, metric=metric)
                        print("calculating meta features...")
                        metaDes.fit(XMeta, YMeta, YCaMeta, folder = folder)

                        for cls in metaClassifiers:
                            metaDes.metaCls = cls
                            name = "metaDes_hC"+str(metaDes.hC)+\
                                   "_K"+str(metaDes.K)+\
                                   "_Kp"+str(metaDes.Kp)+\
                                   "_mode"+metaDes.mode+\
                                   "_competence"+str(metaDes.competenceTresshold)+\
                                   "_cls"+metaDes.metaCls.name+\
                                    "_metric"+metaDes.metric
                            metaDes.fitWithAlreadySaved(saveModel = False, folder = folder) #if we already computed features
                            Helpers.shraniModel(metaDes,folder+"models/"+name+"/"+name) #we save fitted model
                            responseTest = metaDes.predict_proba(XTest, YCaTest, XSel, YSel, YCaSel)[:,1]


                            plotClassifiersAndSaveResult(YTest,YCaTest, responseTest, name, folder=folder) #we save figure and save results
                    except Exception as e:
                        print(str(e))


def plotClassifiersAndSaveResult(YTest, YCaTest, YMetaResponse,graphName, folder = "data/dataForMeta/"):
    #plots all classifiers, that are stored in data/dataForMeta/classifiers, so we can compare precision with respect to
    #meta classifier
    handles = []
    result, hand = dviganjeDecilov(YTest, YMetaResponse, "MetaDes", linewidth=4)
    handles.append(hand)
    clsNames = os.listdir(folder+"classifiers/")
    for i, YCa in enumerate(YCaTest.T):
        handles.append(dviganjeDecilov(YTest, YCa, clsNames[i])[1])
    plt.legend(handles = handles, loc = 2)

    plt.savefig(folder+"graphs/"+graphName+".png", bbox_inches='tight')
    plt.clf()
    print("Result of:"+graphName+":")
    print(str(result))
    pickleListAppend2([result,graphName], folder+"parameterResults.p") #zapisemo rezultat
