__author__ = 'Martin'
import numpy as np
from Helpers import metricWithRawDataAboveDecil
from sklearn.metrics import precision_score
from sklearn.base import clone
"""
    Implementacije metode Bagging. V metodi si izberemo kateri klasifikator bomo uporabljali za osnovni klasifikator,
    Koliko jih bomo ustvarili. Implementirali smo tudi OutOfBag napako, kjer ocenjujemo vsak klasifikator, kaksno
    napako je naredil na testni mnozici. S parametrom samples_ratio dolocimo, kako velika bo ucna mnozica.
    S parametrom ratioCutOffEstimator pa dolocimo, koliko najboljsih klasifikatorjev bomo upostevali.

    Napovedujemo s povprecno vrednostjo izbraznih klasifikatorjev.


    Implementirali smo tudi Bagging Uncertain, ki je podrazred Bagginga, le da pri napovedovanju napovemo primere,
    ki so bili negotovi (torej da so imeli razliko v razlicnih napovedih klasifikatorjev vecjo) in jih napovemo
    kot nedonosne
"""
class Bagging():
    def __init__(self,base_estimator, n_estimators=10, samples_ratio=0.8, ratioCutOffEstimators=0.3):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.samples_ratio = samples_ratio
        self.ratioCutOffEstimators = ratioCutOffEstimators

    def fit(self,X,y):
        self.classes_ = np.unique(y)

        estimators = []
        oobErrors = []
        for i in range(self.n_estimators):
            est = clone(self.base_estimator)
            idxs = np.arange(len(y))
            np.random.shuffle(idxs)
            trainIdx = idxs[:self.samples_ratio*len(y)]
            testIdx = idxs[self.samples_ratio*len(y):]

            est.fit(X[trainIdx], y[trainIdx])
            predProba = est.predict_proba(X[testIdx])[:,1]

            oobErrors.append(metricWithRawDataAboveDecil(predProba,y[testIdx],decil = 0.9, metricFunction=precision_score))
            estimators.append(est)

        self.oobErrorsAndEstimators = list(zip(oobErrors,estimators))
    def predict_proba(self,X):
        cuttedEst = sorted(self.oobErrorsAndEstimators, key=lambda pair:pair[0])[int(self.n_estimators*self.ratioCutOffEstimators):] #we cut bad performing estimators

        proba = np.zeros((X.shape[0], len(self.classes_)))
        for oobErr, est in cuttedEst:
            proba += est.predict_proba(X)
        proba = proba/len(cuttedEst)
        return proba
    def predict(self,X):
        return self.classes_.take(np.argmax(self.predict_proba(X), axis=1),
                                  axis=0)

class BaggingUncertain(Bagging):
    def __init__(self, base_estimator, n_estimators=10, samples_ratio=0.8, ratioCutOffEstimators=0.3, ratioOfDeltaRemove = 0.1):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            samples_ratio=samples_ratio,
            ratioCutOffEstimators=ratioCutOffEstimators
        )
        self.ratioOfDeltaRemove = ratioOfDeltaRemove
    def predict_proba(self,X):
        cuttedEst = sorted(self.oobErrorsAndEstimators, key=lambda pair:pair[0])[int(self.n_estimators*self.ratioCutOffEstimators):] #we cut bad performing estimators

        proba = np.zeros((X.shape[0], len(self.classes_)))

        maxEst = np.zeros((X.shape[0]))
        minEst = np.ones((X.shape[0]))

        for oobErr, est in cuttedEst:
            preds = est.predict_proba(X)
            proba += preds
            maxEst = np.max(np.row_stack([preds[:,1],maxEst]),axis=0)
            minEst = np.min(np.row_stack([preds[:,1],minEst]),axis=0)
        proba = proba/len(cuttedEst)
        delta = maxEst-minEst
        tresshold = sorted(delta)[int(X.shape[0]*(1-self.ratioOfDeltaRemove))]
        idxs = np.where(delta>tresshold)
        # list(zip(delta,proba)).sort()[:X.shape[0]*(1-self.ratioOfDeltaRemove)]
        proba[idxs] = [1,0] #we define that as zero
        return proba
