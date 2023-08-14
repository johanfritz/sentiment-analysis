import json
import numpy as np
import csv
import torch
path="dict.json"
datapath="data/sentiment-emotion-labelled_Dell_tweets.csv"
# with open(path, "w") as file:
#     json.dump(contents, file)
# with open(path, "r") as file:
#     loaded=json.load(file)
sentiment=[]
accuracy=[]
def toInt(text):
    if(text=="negative"):
        return 2
    if(text=="neutral"):
        return 1
    if(text=="positive"):
        return 0
    return -1
with open("indexdict.json") as file:
    indexdict=json.load(file)
# with open(datapath, "r") as file:
#     csvreader=csv.reader(file)
#     for row in csvreader:
#         sentiment.append(toInt(row[5]))
#         accuracy.append(row[6])

# sentences=np.asarray(sentiment)
# accuracy=np.asarray(accuracy)
# np.save("sentiment.npy", sentences)
# np.save("accuracy.npy", accuracy)
# dict={}
# sentences=np.load("sentences.npy")
# for element in sentences:
#     wordlist=words(element).lower().split()
#     for word in wordlist:
#         dict[word]=dict.get(word, 0)+1
# with open(path, "w") as file:
#     json.dump(dict, file)
# actualdict={}
# index=0
# for key in loaded:
#     actualdict[key]=index
#     index+=1
# with open("indexdict.json", "w") as file:
#     json.dump(actualdict, file)
# sentences=np.load("sentences.npy")
# list=[]
# for element in sentences:
#     vector=np.zeros(48554)
#     for word in element.lower().split():
#         if word in indexdict:
#             vector[indexdict[word]]+=1
#     list.append(vector)
# list=np.asarray(list)
# list=torch.from_numpy(list)
# torch.save(list, "sentences.tp")

