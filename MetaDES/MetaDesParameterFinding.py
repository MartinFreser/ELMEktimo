__author__ = 'Martin'
import Helpers
import numpy as np
# from ELMimplementacije.PythonELM.elm import GenELMClassifier
# from ELMimplementacije.PythonELM.random_layer import RandomLayer
from ELMImplementacije.PythonELM.elm import GenELMClassifier
from ELMImplementacije.PythonELM.random_layer import RandomLayer
import os
from MetaDES.MetaDES import MetaDES
from sklearn.naive_bayes import GaussianNB
from MetaDES.HelpersMeta import dviganjeDecilov
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import traceback

from Helpers import pickleListAppend2
import pickle

from sklearn.ensemble import RandomForestClassifier
from testingMetaDes import readForMeta2, readClsResponse
from sklearn.neighbors.ball_tree import BallTree

def findParameters(folder = "data/dataForMeta/ostanek/"):
    """
    V funkciji zaganjamo algoritem MetaDES na diskretnem naboru parametrov. Za vsako iteracijo shranimo zgrajen model,
    shranimo rezultat po decilih in shranimo izrisan graf, ki nam zraven rezultata algoritma MetaDes izrise se rezultat
    vseh algoritmov, ki so bili uporabljeni v ensemblu MetaDES.
    :return:
    """


    XMeta, YMeta, XSel, YSel, XTest, YTest = readForMeta2(folder = folder)

    nb = GaussianNB()#meta classifier for metaDes
    nb.name="Bayes"
    rf = RandomForestClassifier(n_estimators=1000, n_jobs=2)
    rf.name="rf"
    elm = GenELMClassifier(hidden_layer = RandomLayer(n_hidden = 400, activation_func = 'multiquadric', alpha=1))
    elm.name="elm"
    lr = LogisticRegression()
    lr.name= "lr"

    metaClassifiers = [lr, elm]
    hCs = [1.0, 0.5]
    nrNeigh = [1000]#, 1000, 3000]
    modes = ["weighted"]
    metrics = ["l2", "chebyshev"]#BallTree.valid_metrics
    metaClsModes = ["combined"]
    normalizeMetaFeatures = [True, False]
    competenceTressholds = [0.4,0.5,0.6]

    # metaDes = MetaDES(0.8,1000, 50, lr, competenceTresshold=0.5, mode="weightedAll")


    YCaMeta = readClsResponse("Meta", folder=folder) #we read all classifications for meta dataset
    YCaSel = readClsResponse("Sel", folder = folder)
    YCaTest = readClsResponse("Test", folder = folder)

    nrOfTrials = 0
    allTrials = len(nrNeigh)*len(hCs)*len(modes)*len(metrics)*len(metaClassifiers)*len(metaClsModes)*len(normalizeMetaFeatures)
    print("We will have %d trials" %allTrials)
    for nrN in nrNeigh:
        for hC in hCs:
            for mode in modes:
                for metric in metrics:
                    try:
                        metaDes = MetaDES(hC,nrN, nrN, lr, competenceTresshold=0.5, mode=mode,
                                          metric=metric)
                        print("calculating meta features...")
                        metaDes.fit(XMeta, YMeta, YCaMeta, folder = folder)

                        for cls in metaClassifiers:
                            for metaClsMode in metaClsModes:
                                for normalizeMetaFeat in normalizeMetaFeatures:
                                    metaDes.metaClsMode = metaClsMode
                                    metaDes.metaCls = cls
                                    metaDes.normalizeMetaFeat = normalizeMetaFeat
                                    name = "metaDes_hC"+str(metaDes.hC)+\
                                           "_K"+str(metaDes.K)+\
                                           "_Kp"+str(metaDes.Kp)+\
                                           "_mode"+metaDes.mode+\
                                           "_competence"+str(metaDes.competenceTresshold)+\
                                           "_cls"+metaDes.metaCls.name+\
                                            "_metric"+metaDes.metric+\
                                            "_metaClsMode"+metaDes.metaClsMode+\
                                            "_normMetaFeat" + str(metaDes.normalizeMetaFeat)

                                    nrOfTrials += 1
                                    print("Fitting %d/%d trial" %(nrOfTrials,allTrials))

                                    metaDes.fitWithAlreadySaved(saveModel = False, folder = folder) #if we already computed features

                                    responseTest = metaDes.predict_proba(XTest, YCaTest, XSel, YSel, YCaSel)[:,1]


                                    plotClassifiersAndSaveResult(YTest,YCaTest, responseTest, name, folder=folder) #we save figure and save results
                                    Helpers.shraniModel(metaDes,folder+"models/"+name+"/") #we save fitted model
                    except Exception as e:
                        allTrials -= 1
                        with open(folder+"error.log", "a") as fw:
                            fw.write("We were executing "+name+"\n")
                            fw.write(str(traceback.format_exc())+"\n\n\n***************************************")
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

def loadResults(file = "data/dataForMeta/ostanek/parameterResults.p", sortMode = "lastPercentil"):
    """
    Funkcija nalozi rezultate pridobljene v funkciji poisciParametre(), ter jih uredi po nacinu izbranem
    v parametru sortMode.
    :param file:
    :param sortMode:
                "lastPercentil" ... uredi po velikosti zadnjega percentila (95-ega)
                "last4percentils" ... sestej vsoto zadnjih 4 percentilov in uredi po velikosti
                "90percentile" ... uredi po velikosti predzadnjega percentila (90-ega)
                Ce je podan kaksen drug string, uredi po velikosti 80-ega percentila
    :return:
    """
    list = pickle.load(open(file,"rb"))
    sortFun = lambda x: sum(x[0][2:6]) if sortMode == "last4percentils" else x[0][5] if sortMode == "lastPercentil" else x[0][4] if sortMode == "90percentile" else x[0][3]
    list.sort(key = sortFun, reverse=True)
    print("\n".join(map(str,list)))
