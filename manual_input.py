from network import *
import json

with open("indexdict.json", "r") as file:
    indexdict = json.load(file)


neuralnet = model(0, 0)


def toVector(text):
    bow = torch.zeros(48554)
    word = ""
    space = False
    for k in range(len(text)):
        value = ord(text[k])
        if value == 32 and not space:
            word += text[k]
            space = True
        if (value > 64 and value < 91) or (value > 96 and value < 123):
            word += text[k]
            space = False
    for element in word.lower().split():
        if element in indexdict:
            bow[indexdict[element]] += 1
    return bow


def sentiment(a3):
    guess = torch.argmax(a3)
    if guess == 0:
        sentiment = "positive"
    if guess == 1:
        sentiment = "neutral"
    if guess == 2:
        sentiment = "negative"
    text = "sentiment is " + sentiment + ", confidence is " + \
        str(a3[guess].item()) + "      " + str(a3.detach().numpy())
    return text


while True:
    text = input("Enter string:")
    out = toVector(text)
    text = sentiment(neuralnet.forward(out))
    print(text)
