__author__ = 'Martin'
import numpy as np
from MetaDES.HelpersMeta import DOC, computeMetaFeatures
import time
from sklearn.neighbors import NearestNeighbors
import Helpers
from sklearn.externals import joblib
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

"""
    Razred implementira metodo Meta Dynamic Ensemble Selection.
"""

class MetaDES():
    """
    # hc ... consensus treeshold
    # K ... Number of nearest neighbours of Region
    # Kp ... Number of nearest neighbours of Output Region
    # metaCls ... Meta classifier, which decides, if classifiers prediction is competent or not
    # mode ... possible choices are "mean", "majorityVote", "majorityVoteProbs", "weighted", "weightedAll"
            "mean" ... predict mean predictions off all classifiers, who has competence above competenceTresshold
            "majorityVote" ... predict 0 or 1, according to majority votes of competent aclassifiers
            "majorityVoteProbs" ... predict ration between number of classifiers, who predict above 0.5 and number
                off all classifiers
            "weighted" ... predicts weighted sum of predictions of competent classifiers according to their competence
            "weightedAll" ... same as weighted, except it takes into account all classifiers
        competenceTresshold ... tresshold, whether classifiers are competent or not
        metric ... metric to use to measure distance between examples. Should be compatible with kd_tree or Ball_tree
            metrics.
        metaClsMode ... possible choices:
            "one" ... We use one classifier to compute competence
            "combined" ... We use as many meta classifiers as there are classifiers, so we have one metaClassifier for
                each classifier to tell us, wether classifier is competent or not

    """
    def __init__(self, hC, K, Kp, metaCls, nrOfClassifiers = None, mode = "mean",metric = "l2", competenceTresshold = 0.5,
                 metaClsMode = "one", printing = True, normalizeMetaFeat = True):
        self.hC = hC
        self.K = K
        self.Kp = Kp
        self.metaCls = metaCls
        self.nrOfClassifiers = nrOfClassifiers
        self.mode = mode
        self.competenceTresshold = competenceTresshold
        self.metric = metric
        self.printing = printing
        self.metaClsMode = metaClsMode #"one" is if we only use one metaCls, and "combined" is, if we use own meta classifier for every classifier
        self.normalizeMetaFeat = normalizeMetaFeat
        self.regionsFitted = False #if we call predict_proba() more times, we only have to fit region once
    def fit(self, XMeta, YMeta, YCaMeta, folder = "data/dataForMeta/"): #X ... features, y... trueValue, yC ... values predicted by classifier
        self.nrOfClassifiers = YCaMeta.shape[1]
        wholeTime, timeForRegion = 0,0
        start = time.time()
        if self.printing: print("Starting to fit MetaDes")
        metaFeatures = []
        metaResponse = []
        nearestNeigbourRegion = NearestNeighbors(n_neighbors=self.K, metric=self.metric)
        nearestNeigbourRegion.fit(XMeta)
        nearestNeigbourOutputRegion = NearestNeighbors(n_neighbors=self.Kp, metric=self.metric)
        nearestNeigbourOutputRegion.fit(np.round(YCaMeta))
        with open(folder+"MetaFeatures_K"+str(self.K)+"_Kp"+str(self.Kp)+".csv", "w") as fMetaFeatures:
            #we use this, because this can be very big folder, so we have to save incrementally in file
            for i, x in enumerate(XMeta):
            # for i in range(2000):
                x = XMeta[i]
                if(i%1000 == 0): print("Training examples covered: %d/%d" %(i, len(XMeta)))
                doc = DOC(np.round(YCaMeta[i]), mode=1)#degree of consensus, Morda premislit, kako to drugace dolocit
                if(doc <= self.hC): #we let in instances, where classifiers have smaller consensus than tresshold
                    reg, opReg = {},{}
                    start2 = time.time()
                    # idxsReg = findRegion(XMeta, x, self.K, method='normalRegion')
                    idxsReg = nearestNeigbourRegion.kneighbors(x, n_neighbors=self.K+1, return_distance=False)[0,1:]
                    timeForRegion+= time.time() - start2
                    reg["X"], reg["Y"] = XMeta[idxsReg], YMeta[idxsReg]

                    start2 = time.time()
                    #idxsOP = findRegion(np.round(YCaMeta), np.round(YCaMeta[i]), self.Kp, method='outputProfileRegion')
                    idxsOP = nearestNeigbourOutputRegion.kneighbors(np.round(YCaMeta[i]), n_neighbors=self.Kp + 1, return_distance=False)[0,1:]
                    timeForRegion += time.time() - start2

                    opReg["X"], opReg["Y"] = XMeta[idxsOP], YMeta[idxsOP]
                    for j, cls in enumerate(YCaMeta[i]):
                        reg["YC"] = YCaMeta[idxsReg][:,j] #vzamemo vse response j-tega classifierja v okolici x
                        opReg["YC"] = YCaMeta[idxsOP][:,j]
                        f = computeMetaFeatures(reg, opReg)
                        metaFeatures.append(list(f))
                        res = 1 if int(np.round(cls)) == int(np.round(YMeta[i])) else 0
                        metaResponse.append(res)
                        [(fMetaFeatures.write(str(feat)), fMetaFeatures.write(",") if i != len(f)-1 else None) for i, feat in enumerate(f)]
                        fMetaFeatures.write("\n")
        metaResponse = np.array(metaResponse)
        np.savetxt(folder+"MetaResponse_K"+str(self.K)+"_Kp"+str(self.Kp)+".csv", metaResponse, delimiter="\n")
        metaFeatures = np.array(metaFeatures)

        print("Fitting meta cls...")
        self.fitMetaCls(metaFeatures, metaResponse)
        print("Done!")

        wholeTime = time.time()-start
        print("For training metaDes we needed %d time for finding region out of %d \n "
              "so we spent %.3f for region seeking" %(timeForRegion, wholeTime, timeForRegion/wholeTime))
    def fitMetaCls(self,metaFeatures, metaResponse):
        #Method fits metaClassifier

        #scale metaFeatures
        if self.normalizeMetaFeat:
            self.scaler = StandardScaler()
            self.scaler.fit(metaFeatures)
            metaFeatures = self.scaler.transform(metaFeatures)
        if self.metaClsMode == "one":
            self.metaCls.fit(metaFeatures, metaResponse)
        elif self.metaClsMode == "combined":
            #Matriko razdelimo na nrOfClassifiers matrik
            self.metaClassifiers = [clone(self.metaCls) for i in range(self.nrOfClassifiers)]
            sanityCheck = 0
            for i in range(self.nrOfClassifiers):
                features = metaFeatures[i:][::self.nrOfClassifiers]
                responses = metaResponse[i:][::self.nrOfClassifiers]
                self.metaClassifiers[i].fit(features,responses)
                sanityCheck+=features.shape[0]
            if(sanityCheck != metaFeatures.shape[0]): raise AttributeError("metaFeatures are not dividable with nrOfClassifiers")
    def fitWithAlreadySaved(self, saveModel = True, folder = "data/dataForMeta/"):
        print("reading already saved features in "+folder)
        metaFeatures = np.loadtxt(folder + "MetaFeatures_K"+str(self.K)+"_Kp"+str(self.Kp)+".csv", delimiter=",")
        metaResponse = np.loadtxt(folder + "MetaResponse_K"+str(self.K)+"_Kp"+str(self.Kp)+".csv", delimiter = "\n")
        print("We are fitting already computed features in "+folder)
        print("shape of the feature matrix: " + str(metaFeatures.shape))
        self.fitMetaCls(metaFeatures,metaResponse)
        print("done!")
        if saveModel: Helpers.shraniModel(self.metaCls, "data/dataForMeta/metaModels/"+self.metaCls.name+"/"+self.metaCls.name+".p")
    def loadMetaCls(self):
        cls = joblib.load("data/dataForMeta/metaModels/"+self.metaCls.name+"/"+self.metaCls.name+".p")
        self.cls = cls

    # @profile #we want to measure time for our method
    def predict_proba(self, XTest, YCaTest, XSel, YSel, YCaSel):
        wholeTime, timeForRegion, timeForCls = 0,0,0
        start = time.time()
        response = []

        if(not self.regionsFitted):
            nearestNeigbourRegion = NearestNeighbors(n_neighbors=self.K, metric=self.metric)
            nearestNeigbourRegion.fit(XSel)
            self.nearestNeigbourRegion = nearestNeigbourRegion
            nearestNeigbourOutputRegion = NearestNeighbors(n_neighbors=self.Kp, metric=self.metric)
            nearestNeigbourOutputRegion.fit(np.round(YCaSel))
            self.nearestNeigbourOutputRegion = nearestNeigbourOutputRegion
            self.regionsFitted = True

        for i, x in enumerate(XTest):
            if(i%1000 == 0): print("Test examples covered: %d/%d" %(i, len(XTest)))
            reg, opReg = {},{}

            start2 = time.time()
            #idxsReg = findRegion(XSel, x, self.K, method='normalRegion')
            idxsReg = self.nearestNeigbourRegion.kneighbors(x, n_neighbors=self.K, return_distance=False)[0]
            # idxsReg = range(self.K)
            timeForRegion+= (time.time() - start2)
            reg["X"], reg["Y"] = XSel[idxsReg], YSel[idxsReg]

            start2 = time.time()
            # idxsOP = findRegion(np.round(YCaSel), np.round(YCaTest[i]), self.K, method='outputProfileRegion')
            idxsOP = self.nearestNeigbourOutputRegion.kneighbors(np.round(YCaTest[i]), n_neighbors=self.Kp, return_distance=False)[0]
            # idxsOP = range(self.Kp)
            timeForRegion+= (time.time() - start2)
            opReg["X"], opReg["Y"] = XSel[idxsOP], YSel[idxsOP]
            votes = []
            competenceOfClassifiers = []
            start3 = time.time()
            selectedCls = []
            allCls = []
            metaFeatures = []
            for j, cls in enumerate(YCaTest[i]):
                reg["YC"] = YCaSel[idxsReg][:,j] #vzamemo vse response classifierja v okolici x
                opReg["YC"] = YCaSel[idxsOP][:,j]
                metaFeatures.append(computeMetaFeatures(reg, opReg))

            competence = self.predictWithMetaCls(metaFeatures)
            allCls = np.column_stack([competence,range(len(YCaTest[i])), YCaTest[i]])
            timeForCls += time.time()-start3
            allCls = allCls[allCls[:,0].argsort()] # we sort after first component
            if(self.mode == "weightedAll"):
                selection = allCls
            else:
                selection = allCls[np.where(allCls[:,0] >= self.competenceTresshold)[0]]
            # if(i%1000 == 1): print("best competence of some case: %.4f", str(allCls[-1,0]))
            if(len(selection) == 0):
                selection=allCls[::-1][:2] #we selcted best two classifiers
                print("we havent found good cls, using best 2 out of bad classifiers, "
                      "competence of the best is: %.4f" %(selection[0,0]))
            votes = selection[:,2] #YCaTest
            competenceOfClassifiers = selection[:,0] #competence
            votes = np.array(votes)
            if self.mode == "mean":
                pred = np.mean(votes) #Uporabimo mean, ali bi se res odlocili za majority vote
            elif self.mode == "majorityVote":
                pred = 1 if len(np.where(votes>0.5)[0]) >= len(votes)/2 else 0
            elif self.mode == "majorityVoteProbs":
                pred = len(np.where(votes>0.5)[0])/len(votes)
            elif self.mode == "weighted" or self.mode == "weightedAll":
                #normalize competence vector so it sums to one
                competenceOfClassifiers = competenceOfClassifiers/sum(competenceOfClassifiers)
                pred = np.sum(competenceOfClassifiers*votes)

            response.append([1-pred,pred])
        wholeTime = time.time() - start
        print("For training metaDes we needed %d time for finding region, %d time for cls out of %d \n "
              %(timeForRegion, timeForCls, wholeTime))
        return np.array(response)
    def predictWithMetaCls(self,metaFeatures):
        if self.normalizeMetaFeat:
            metaFeatures = self.scaler.transform(metaFeatures)
        if self.metaClsMode == "one":
            return self.metaCls.predict_proba(metaFeatures)[:,1]
        elif self.metaClsMode == "combined":
            responses = np.array([self.metaClassifiers[i].predict_proba(metaFeatures[i])[0] for i in range(self.nrOfClassifiers)])
            return responses[:,1]
    def predict(self,XTest, XSel, YSel, YCaSel):
        return self.classes_.take(np.argmax(self.predict_proba(XTest, XSel, YSel, YCaSel), axis=1),
                                  axis=0)









