__author__ = 'Martin'
import numpy as np
def maliKos():
    with open("//./Z:/Podatki/Prediction datasets/fullTestFeatures.csv") as f:
        i = 0
        f.readline() #dummy read for header
        with open("//./Z:/Martin Freser/ELMEktimo/data/dataForMeta/ostanek/fullTestFeatures.csv", "w") as f2:
            for line in f:
                i+=1
                content = np.array(line.replace("\"", "").split(",")[2:]).astype(float)
                content = ",".join(map(str,list(content)))
                f2.write(content+"\n")
                if(i%100000 == 0): print("%d/%d" %(i, 1000000))
        print("we wrote %d lines" %i)


if __name__ == "__main__":
    maliKos()