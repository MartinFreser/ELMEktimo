setwd("D:\\Martin delo\\Ektimo\\ELM\\ELMEktimo")
smallFtr <- readRDS("data/smallTrainFeatures.RDS")
smallCls <- readRDS("data/smallTrainClassifier.RDS")
write.csv(smallFtr,file = "smallTrainFeatures.csv")
write.csv(smallCls,file = "smallTrainClassifier.csv")
max(smallFtr)

for (i in 3:192){
	print(c(i, mean(smallFtr[,i])))
}

plot(smallFtr[18])

load("data/trainCls.rdata")
load("data/trainFtr.rdata")
load("data/trainSez.rdata")

load("data/testCls.rdata")
load("data/testFtr.rdata")

trainCls = ifelse(trainCls == "G", 1, 0)
write.csv(trainCls, file="data/trainCls.csv", row.names=FALSE)

testCls = ifelse(testCls, 1, 0) #testCls je true/false
write.csv(testCls, file="data/testCls.csv", row.names=FALSE)

write.csv(trainFtr, file="data/trainFtr.csv")
write.csv(testFtr, file="data/testFtr.csv")

