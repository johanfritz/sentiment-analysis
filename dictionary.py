import json
import numpy as np
import csv
import torch

# vector = np.zeros(48554)
# data = np.load("newsentences.npy")
# with open("indexdict.json", "r") as file:
#     indexdict = json.load(file)
# for word in data[2].split():
#     vector[indexdict.get(word)] += 1


def xavier_init(n_in, n_out):
    xavier_stddev = np.sqrt(2.0 / (n_in + n_out))
    return np.random.normal(0, xavier_stddev, (n_out, n_in))


def newweights():
    w0 = xavier_init(48554, 18)
    w1 = xavier_init(18, 16)
    w2 = xavier_init(16, 3)
    b0 = np.ones(18)*0.01
    b1 = np.ones(16)*0.01
    b2 = np.ones(3)*0.01
    np.save('weights-and-biases/w0.npy', w0)
    np.save('weights-and-biases/w1.npy', w1)
    np.save('weights-and-biases/w2.npy', w2)
    np.save('weights-and-biases/b0.npy', b0)
    np.save('weights-and-biases/b1.npy', b1)
    np.save('weights-and-biases/b2.npy', b2)


# newweights()
leaky_const = 0


def scalarrelu(n):
    if (n > 0):
        return n
    return n*leaky_const


relu = np.vectorize(scalarrelu)


def scalarreluprime(n):
    if (n > 0):
        return 1
    return leaky_const


reluPrime = np.vectorize(scalarreluprime)
def sigmoid(x): return 1/(1+np.exp(-x))
def sigmoidPrime(x): return np.exp(-x)/((1+np.exp(-x))**2)


class network:
    # gällande indexering:
    # [0] är viktmatriser
    # [1] är gradientmatriser
    # [2] är momentmatriser
    def __init__(self):
        self.beta = 0.85
        self.iterations = 0
        w0 = [np.load('weights-and-biases/w0.npy')]
        w1 = [np.load('weights-and-biases/w1.npy')]
        w2 = [np.load('weights-and-biases/w2.npy')]
        b0 = [np.load('weights-and-biases/b0.npy')]
        b1 = [np.load('weights-and-biases/b1.npy')]
        b2 = [np.load('weights-and-biases/b2.npy')]
        tempw0 = np.zeros(np.shape(w0[0])).astype(np.float64)
        tempw1 = np.zeros(np.shape(w1[0])).astype(np.float64)
        tempw2 = np.zeros(np.shape(w2[0])).astype(np.float64)
        tempb0 = np.zeros(np.shape(b0[0])).astype(np.float64)
        tempb1 = np.zeros(np.shape(b1[0])).astype(np.float64)
        tempb2 = np.zeros(np.shape(b2[0])).astype(np.float64)
        for k in range(1):
            w0.append(tempw0)
            w1.append(tempw1)
            w2.append(tempw2)
            b0.append(tempb0)
            b1.append(tempb1)
            b2.append(tempb2)
        w0.append(np.load('momentum/w0.npy'))
        w1.append(np.load('momentum/w1.npy'))
        w2.append(np.load('momentum/w2.npy'))
        b0.append(np.load('momentum/b0.npy'))
        b1.append(np.load('momentum/b1.npy'))
        b2.append(np.load('momentum/b2.npy'))
        w0 = np.asarray(w0)
        w1 = np.asarray(w1)
        w2 = np.asarray(w2)
        b0 = np.asarray(b0)
        b1 = np.asarray(b1)
        b2 = np.asarray(b2)
        self.w0 = torch.from_numpy(w0)
        self.w1 = torch.from_numpy(w1)
        self.w2 = torch.from_numpy(w2)
        self.b0 = torch.from_numpy(b0)
        self.b1 = torch.from_numpy(b1)
        self.b2 = torch.from_numpy(b2)

    def save(self):
        np.save('weights-and-biases/w0.npy', self.w0[0])
        np.save('weights-and-biases/w1.npy', self.w1[0])
        np.save('weights-and-biases/w2.npy', self.w2[0])
        np.save('weights-and-biases/b0.npy', self.b0[0])
        np.save('weights-and-biases/b1.npy', self.b1[0])
        np.save('weights-and-biases/b2.npy', self.b2[0])
        np.save('momentum/w0.npy', self.w0[2])
        np.save('momentum/w1.npy', self.w1[2])
        np.save('momentum/w2.npy', self.w2[2])
        np.save('momentum/b0.npy', self.b0[2])
        np.save('momentum/b1.npy', self.b1[2])
        np.save('momentum/b2.npy', self.b2[2])

    def update(self):
        if self.iterations != 0:
            self.w0[0] -= self.w0[1]/self.iterations
            self.w1[0] -= self.w1[1]/self.iterations
            self.w2[0] -= self.w2[1]/self.iterations
            self.b0[0] -= self.b0[1]/self.iterations
            self.b1[0] -= self.b1[1]/self.iterations
            self.b2[0] -= self.b2[1]/self.iterations
            self.iterations = 0

    def descent(self, bow, sentiment, learningrate, confidence):
        cv = torch.zeros([3])
        cv[sentiment] = 1
        # a1, a2, a3, z1, z2, z3 = self.activations(bow)
        z1 = torch.mm(self.w0[0], bow)+self.b0[0]
        a1 = torch.relu(z1)
        z2 = torch.mm(self.w1[0], a1) + self.b1[0]
        a2 = torch.relu(z2)
        z3 = torch.mm(self.w2[0], a2) + self.b2[0]
        a3 = torch.sigmoid(z3)
        z1prime = torch.where(z1, torch.ones(
            [len(z1)]), torch.zeros([len(z1)]))  # reluprime
        z2prime = torch.where(z2, torch.ones(
            [len(z2)]), torch.zeros([len(z2)]))  # reluprime
        z3prime = torch.sigmoid(
            z3)*(torch.ones_like(z3) - torch.sigmoid(z3))  # sigmoidprime
        delta1 = a3-cv
        delta1 = delta1*2*z3prime  # här är jag
        delta2 = delta1.view((torch.size(self.b2[0])[0], 1))*a2
        delta3= torch.mm(self.w2[0].transpose, delta1)*z2prime
        delta4=delta3.view((torch.size(self.b1[0])[0], 1))*a1
        delta5=torch.mm(self.w1[0].transpose, delta3)*z1prime
        delta6 = delta5.reshape((np.shape(self.b0[0])[0], 1))*bow
        delta6=delta5.view((torch.size(self.b0[0])[0], 1))*bow
        self.w0[2] = self.beta*self.w0[2]+(1-self.beta)*delta6
        self.w1[2] = self.beta*self.w1[2]+(1-self.beta)*delta4
        self.w2[2] = self.beta*self.w2[2]+(1-self.beta)*delta2
        self.b0[2] = self.beta*self.b0[2]+(1-self.beta)*delta5
        self.b1[2] = self.beta*self.b1[2]+(1-self.beta)*delta3
        self.b2[2] = self.beta*self.b2[2]+(1-self.beta)*delta1
        self.w0[1] += self.w0[2]*learningrate
        self.w1[1] += self.w1[2]*learningrate
        self.w2[1] += self.w2[2]*learningrate
        self.b0[1] += self.b0[2]*learningrate
        self.b1[1] += self.b1[2]*learningrate
        self.b2[1] += self.b2[2]*learningrate
        self.iterations += 1
        return 0, torch.argmax(a3)

    def activations(self, bow):
        z1 = np.matmul(self.w0[0], bow) + self.b0[0]
        a1 = relu(z1)
        # print("a1" + str(type(a1[0])))
        z2 = np.matmul(self.w1[0], a1) + self.b1[0]
        a2 = relu(z2)
        # print("a2" + str(type(a2[0])))
        z3 = np.matmul(self.w2[0], a2) + self.b2[0]
        a3 = sigmoid(z3)
        return a1, a2, a3, z1, z2, z3

    def forward(self, bow):
        z1 = np.matmul(self.w0[0], bow) + self.b0[0]
        a1 = relu(z1)
        z2 = np.matmul(self.w1[0], a1) + self.b1[0]
        a2 = relu(z2)
        z3 = np.matmul(self.w2[0], a2) + self.b2[0]
        a3 = sigmoid(z3)
        return a3
