__author__ = 'Martin'
def maliKos():
    with open("//./Z:/Podatki/Prediction datasets/fullTestFeatures.csv") as f:
        with open("data/fullTestFeaturesMaliKos.csv", "w") as f2:
            for i in range(2000):
                f2.write(f.readline())