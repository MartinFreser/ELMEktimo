__author__ = 'Martin'
import numpy as np
from MetaDES.HelpersMeta import findRegion, DOC, computeMetaFeatures
import time
from sklearn.neighbors import NearestNeighbors
import Helpers
from sklearn.externals import joblib

class MetaDES():
    # hc ... consensus treeshold
    def __init__(self, hC, K, Kp, metaCls, mode = "mean",metric = "l2", competenceTresshold = 0.5, printing = True):
        self.hC = hC
        self.K = K
        self.Kp = Kp
        self.metaCls = metaCls
        self.mode = mode
        self.competenceTresshold = competenceTresshold
        self.metric = metric
        self.printing = printing
    def fit(self, XMeta, YMeta, YCaMeta, folder = "data/dataForMeta/"): #X ... features, y... trueValue, yC ... values predicted by classifier
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
                doc = DOC(np.round(YCaMeta[i]))#degree of consensus, Morda premislit, kako to drugace dolocit
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
                        reg["YC"] = YCaMeta[idxsReg][:,j] #vzamemo vse response classifierja v okolici x
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
        self.metaCls.fit(metaFeatures, metaResponse)
        print("Done!")

        wholeTime = time.time()-start
        print("For training metaDes we needed %d time for finding region out of %d \n "
              "so we spent %.3f for region seeking" %(timeForRegion, wholeTime, timeForRegion/wholeTime))
    def fitWithAlreadySaved(self, saveModel = True, folder = "data/dataForMeta/"):
        print("reading already saved features in "+folder)
        metaFeatures = np.loadtxt(folder + "MetaFeatures_K"+str(self.K)+"_Kp"+str(self.Kp)+".csv", delimiter=",")
        metaResponse = np.loadtxt(folder + "MetaResponse_K"+str(self.K)+"_Kp"+str(self.Kp)+".csv", delimiter = "\n")
        print("We are fitting already computed features in "+folder)
        print("shape of the feature matrix: " + str(metaFeatures.shape))
        self.metaCls.fit(metaFeatures, metaResponse)
        print("done!")
        if saveModel: Helpers.shraniModel(self.metaCls, "data/dataForMeta/metaModels/"+self.metaCls.name+"/"+self.metaCls.name+".p")
    def loadMetaCls(self):
        cls = joblib.load("data/dataForMeta/metaModels/"+self.metaCls.name+"/"+self.metaCls.name+".p")
        self.cls = cls

    # @profile
    def predict_proba(self, XTest, YCaTest, XSel, YSel, YCaSel):
        wholeTime, timeForRegion, timeForCls = 0,0,0
        start = time.time()
        response = []

        nearestNeigbourRegion = NearestNeighbors(n_neighbors=self.K, metric=self.metric)
        nearestNeigbourRegion.fit(XSel)
        nearestNeigbourOutputRegion = NearestNeighbors(n_neighbors=self.Kp, metric=self.metric)
        nearestNeigbourOutputRegion.fit(np.round(YCaSel))

        for i, x in enumerate(XTest):
            if(i%1000 == 0): print("Test examples covered: %d/%d" %(i, len(XTest)))
            reg, opReg = {},{}

            start2 = time.time()
            #idxsReg = findRegion(XSel, x, self.K, method='normalRegion')
            idxsReg = nearestNeigbourRegion.kneighbors(x, n_neighbors=self.K, return_distance=False)[0]
            # idxsReg = range(self.K)
            timeForRegion+= (time.time() - start2)
            reg["X"], reg["Y"] = XSel[idxsReg], YSel[idxsReg]

            start2 = time.time()
            # idxsOP = findRegion(np.round(YCaSel), np.round(YCaTest[i]), self.K, method='outputProfileRegion')
            idxsOP = nearestNeigbourOutputRegion.kneighbors(np.round(YCaTest[i]), n_neighbors=self.Kp, return_distance=False)[0]
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
            competence = self.metaCls.predict_proba(metaFeatures)[:,1]
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
    def predict(self,XTest, XSel, YSel, YCaSel):
        return self.classes_.take(np.argmax(self.predict_proba(XTest, XSel, YSel, YCaSel), axis=1),
                                  axis=0)









