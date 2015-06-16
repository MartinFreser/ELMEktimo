setwd("//./Z:/Podatki/Prediction datasets/")

data = readRDS("ostanekTrainFeatures.RDS")

setwd("D:/Martin delo/Ektimo/ELM/ELMEktimo/data")
write.csv(data, "ostanekTrainFeaturesWithTickers.csv", row.names = FALSE)