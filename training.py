from dictionary import *
import matplotlib.pyplot as plt
import torch

neuralnet = network()
lr = 1e-5
sentences, sentiment, confidence = np.load("newsentences.npy"), np.load(
    "sentiment.npy"), np.load("confidence.npy")

with open("indexdict.json", "r") as file:
    indexdict = json.load(file)

total, correct = 0, 0
totalTotal, correctTotal = 0, 0
list = []
localAccuracy = {0: 0, 1: 0, 2: 0}
localTotal = {}

for k in range(3000):
    # rand = np.random.randint(20000, 24970)
    rand = np.random.randint(2, 20000)
    bow = torch.zeros((48554, 1))
    for word in sentences[rand].split():
        if word in indexdict:
            bow[indexdict[word]] += 1
    norm, guess = neuralnet.descent(bow, sentiment[rand], lr, confidence[rand])
    if (guess == sentiment[rand]):
        localAccuracy[guess] = localAccuracy.get(guess, 0)+1
        correct += 1
        correctTotal += 1
    total += 1
    totalTotal += 1
    localTotal[sentiment[rand]] = localTotal.get(sentiment[rand], 0)+1
    neuralnet.update()
    if (k % 100 == 0) and k != 0:
        print(correct/total)
        list.append(correct/total)
        total, correct = 0, 0
    # print(norm)
#neuralnet.save()
print("done")
print(localAccuracy[0]/localTotal[0])
print(localAccuracy[1]/localTotal[1])
print(localAccuracy[2]/localTotal[2])
print("total")
print(correctTotal/totalTotal)
list = np.asarray(list)
x = np.linspace(0, len(list), len(list))
plt.plot(x, list)
plt.ylim(0, 1)
plt.show()
