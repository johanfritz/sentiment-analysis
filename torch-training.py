from network import *
import json
import numpy as np
lr = 1e-4
mom = 0.9
trainSet = torch.load("torch_data.pth")
neuralnet = model(lr, mom)
sentiment = torch.load("sentiment.pth")
localCorrect = {0: 0, 1: 0, 2: 0}
localTotal = {0: 0, 1: 0, 2: 0}
total=0
totalCorrect=0
for k in range(10000):
    rand = np.random.randint(2, 20000)
    loss, corr = neuralnet.train(trainSet[rand], sentiment[rand], k)
    localTotal[sentiment[rand].item()]=localTotal.get(sentiment[rand].item(), 0) +1
    total+=1
    if corr:
        localCorrect[sentiment[rand].item()]=localCorrect.get(sentiment[rand].item(), 0)+1
        totalCorrect+=1
    if k % 100 == 0:
        print(loss)
#neuralnet.save()
print(localCorrect[0]/localTotal[0])
print(localCorrect[1]/localTotal[1])
print(localCorrect[2]/localTotal[2])
print(totalCorrect/total)