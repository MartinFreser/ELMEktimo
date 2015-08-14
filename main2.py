import MetaDES.MetaDesParameterFinding
import MetaDES.HelpersZaOstanek
import scipy
import numpy as np
import sklearn
if __name__ == "__main__":

    # MetaDES.HelpersZaOstanek.rezreziFileProces()
    MetaDES.MetaDesParameterFinding.findParameters()
    # MetaDES.MetaDesParameterFinding.loadResults(sortMode="90percentile")
    # print("hi")

    # X = np.loadtxt("data/dataForMeta/ostanek/fullTestFeatures.csv")
