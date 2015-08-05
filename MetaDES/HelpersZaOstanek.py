__author__ = 'Martin'
import random, numpy as np, os
import MetaDES.HelpersMeta as hm
from sklearn.externals import joblib
from testingMetaDes import readClsResponse, readForMeta3

def generirajIndexeZaOstanek(folder, nrOfInstances = 100000):
    idxMeta = set(random.sample(range(300000), nrOfInstances))
    idxSel = set(random.sample(range(300000, 600000), nrOfInstances))
    idxTest = set(random.sample(range(600000,900000), nrOfInstances))
    np.savetxt(folder+"idxMeta.csv",np.sort(np.array(list(idxMeta))), delimiter="\n", fmt = "%d")
    np.savetxt(folder+"idxSel.csv",np.sort(np.array(list(idxSel))), delimiter="\n", fmt = "%d")
    np.savetxt(folder+"idxTest.csv",np.sort(np.array(list(idxTest))), delimiter="\n", fmt = "%d")
def razreziFileZaMetaX(bigFileX = "data/dataForMeta/ostanek/ostanekTrainFeatures.csv",folder="data/dataForMeta/ostanek/"):
    #zapisemo prvih nrOfInstances v Meta in drugih nrOfInstances v Selection, tretji pa v Test
    idxMeta = np.loadtxt(folder+"idxMeta.csv", delimiter="\n").astype(int)
    idxSel = np.loadtxt(folder+"idxSel.csv", delimiter="\n").astype(int)
    idxTest = np.loadtxt(folder+"idxTest.csv", delimiter="\n").astype(int)
    checkDistinctOfIdx(idxMeta, idxSel, idxTest)
    fXMeta = open(folder+"XMeta.csv", "w")
    fXSel = open(folder+"XSel.csv", "w")
    fXTest = open(folder+"XTest.csv", "w")
    metaCount, selCount, testCount = 0,0,0
    with open(bigFileX) as f:
        f.readline() #dummy line, header
        i = -1
        for line in f:
            if(i%50000 == 0): print("%d/1000000" %i)
            i+=1
            if(len(line.split(","))<190):
                print(line)
                print("it is something wrong with this line!, this is %d th line" %i)
            if i in idxMeta:
                metaCount+=1
                fXMeta.write(line) #XMeta
            elif i in idxSel:
                selCount+=1
                fXSel.write(line) #XSel
            elif i in idxTest:
                testCount+=1
                fXTest.write(line) #XTest
    print("done with cutting, i wrote %d in XMeta, %d in XSel, and %d in XTest" %(metaCount, selCount, testCount))
    fXMeta.close()
    fXSel.close()
    fXTest.close()
def checkDistinctOfIdx(idxMeta,idxSel,idxTest):
    distMeta = len(set(idxMeta))
    distSel = len(set(idxSel))
    distTest = len(set(idxTest))

    print(distMeta, distSel, distTest)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def popraviClsResponse(file):
    with open(file) as f:
        for line in f:
            pass



def popraviOstanekCls():
    with open("//./Z:/Podatki/Prediction datasets/ostanekTrainClassifier.csv") as f:
        f.readline()
        response = []
        for line in f:
            response.append(1 if "TRUE" in line else 0)
    np.savetxt("data/dataForMeta/ostanek/ostanekResponse.csv",np.array(response),delimiter="\n", fmt = "%d")
def razreziY(bigFileY, folder="data/dataForMeta/ostanek/"):
    idxMeta = np.loadtxt(folder+"idxMeta.csv", delimiter="\n").astype(int)
    idxSel = np.loadtxt(folder+"idxSel.csv", delimiter="\n").astype(int)
    idxTest = np.loadtxt(folder+"idxTest.csv", delimiter="\n").astype(int)

    print(len(set(idxMeta)), len(set(idxSel)), len(set(idxTest)))
    Y = np.loadtxt(bigFileY, delimiter="\n")
    YMeta = Y[idxMeta]
    YSel = Y[idxSel]
    YTest = Y[idxTest]
    np.savetxt(folder+"YMeta.csv", YMeta, fmt = "%d")
    np.savetxt(folder+"YSel.csv", YSel, fmt = "%d")
    np.savetxt(folder+"YTest.csv", YTest, fmt = "%d")

def razreziResponseZaMeta(folderToSave, responseFile, idxMeta = None, idxSel = None, idxTest = None):
    if(idxMeta is None or idxTest is None or idxSel is None):
        idxMeta = np.loadtxt("data/dataForMeta/ostanek/idxMeta.csv", delimiter="\n").astype(int)
        idxSel = np.loadtxt("data/dataForMeta/ostanek/idxSel.csv", delimiter="\n").astype(int)
        idxTest = np.loadtxt("data/dataForMeta/ostanek/idxTest.csv", delimiter="\n").astype(int)

    Y = np.loadtxt(responseFile, delimiter="\n", skiprows=1)
    YMeta = Y[idxMeta]
    YSel = Y[idxSel]
    YTest = Y[idxTest]
    np.savetxt(folderToSave+"YCaMeta.csv", YMeta,delimiter= "\n", fmt = "%.5f")
    np.savetxt(folderToSave+"YCaSel.csv", YSel, delimiter= "\n", fmt = "%.5f")
    np.savetxt(folderToSave+"YCaTest.csv", YTest, delimiter= "\n", fmt = "%.5f")

def razreziClassifierje(folderOfClassifiers = "data/dataForMeta/ostanek/JureClassifiers/Podatki/Classifiers/",
                        folderToSaveCuttedCls = "data/dataForMeta/ostanek/classifiers/"):
    idxMeta = np.loadtxt("data/dataForMeta/ostanek/idxMeta.csv", delimiter="\n").astype(int)
    idxSel = np.loadtxt("data/dataForMeta/ostanek/idxSel.csv", delimiter="\n").astype(int)
    idxTest = np.loadtxt("data/dataForMeta/ostanek/idxTest.csv", delimiter="\n").astype(int)


    for fCls in os.listdir(folderOfClassifiers):
        print("classifier: "+fCls)
        if(not os.path.exists(folderToSaveCuttedCls+fCls.split(".")[0]+"/")): os.makedirs(folderToSaveCuttedCls+fCls.split(".")[0]+"/")
        razreziResponseZaMeta(folderToSaveCuttedCls+fCls.split(".")[0]+"/",folderOfClassifiers+fCls,
                              idxMeta, idxSel, idxTest)

def rezreziFileProces():
    #Metoda doloci indekse za Meta, Selection in Test, katere bomo vzeli iz mnozice ostanekTrain
    folder = "data/dataForMeta/ostanek/"
    generirajIndexeZaOstanek(folder, nrOfInstances=300000) #se izvede samo enkrat, da generiramo vse indexe

    bigFileX = "data/dataForMeta/ostanek/ostanekTrainFeatures.csv"
    bigFileY = "data/dataForMeta/ostanek/ostanekTrainResponse.csv"
    razreziFileZaMetaX()
    razreziY(bigFileY)
    razreziClassifierje()

def fullTestProcess(folder = "data/dataForMeta/ostanek/", clsFile = "data/dataForMeta/models/metaDes/metaDes.p",
                    XTestFile = "data/dataForMeta/ostanek/XTest.csv",
                    YCaTestFolder = "data/dataForMeta/ostanek/",
                    step = 80000):
    #pri metodi bo treba inkrementalno vzeti primere in jih predictat...
    XMeta, YMeta, XSel, YSel = readForMeta3(folder = folder)
    metaDes = joblib.load(clsFile)


    YCaSel = readClsResponse("Sel", folder = folder)
    YCaTest = readClsResponse("Test", folder = YCaTestFolder) #Tu morem dat kazalec na folder, kjer bojo responsi...

    with open(XTestFile) as f:
        i = 0
        XTest = []
        responseTest = []
        for line in f:
            i += 1
            XTest.append(line.strip().split(","))
            if(i%step == 0 and i != 0):
                XTest = np.array(XTest).astype(float)
                YCaTest2 = YCaTest[i-step:i,:]
                responseTest += list(metaDes.predict_proba(XTest, YCaTest2, XSel, YSel, YCaSel)[:,1])

                XTest = []
    if(len(XTest) != 0):#izpraznimo buffer
        XTest = np.array(XTest).astype(float)
        YCaTest2 = YCaTest[i-step:i,:]
        responseTest += list(metaDes.predict_proba(XTest, YCaTest2, XSel, YSel, YCaSel)[:,1])
    responseTest = np.array(responseTest)
    np.savetxt(folder + "MetaDesResponse_"+metaDes.mode+".csv",responseTest, delimiter="\n")