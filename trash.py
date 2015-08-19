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

def maloPreberi():
    with open("data/dataForMeta/ostanek/fullTestBrezCudni.csv") as f:
        i = 0

        for line in f:
            print(line)
            print(len(line.split(",")))
            # print(max(map(float,line.split(","))))
            if (i>4): break
            i+=1
if __name__ == "__main__":
    maloPreberi()