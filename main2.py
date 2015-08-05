import MetaDES.MetaDesParameterFinding
import MetaDES.HelpersZaOstanek
import scipy
import numpy as np
import sklearn
if __name__ == "__main__":

    # MetaDES.MetaDesParameterFinding.findParameters()
    MetaDES.MetaDesParameterFinding.loadResults(sortMode="90percentile")
    # MetaDES.HelpersZaOstanek.rezreziFileProces()
    # print("hi")