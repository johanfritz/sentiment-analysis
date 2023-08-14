import torch
import torch.nn as nn
model = nn.Sequential(
    nn.Linear(48554, 18),
    nn.ReLU(),
    nn.Linear(18, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
    nn.Sigmoid()
)
# torch.save(model, "layers/model.pth")


class model(nn.Module):
    def __init__(self, lr, mom):
        super().__init__()
        self.model = torch.load("layers/model.pth")
        self.optim = torch.optim.SGD(
            self.model.parameters(), lr=lr, momentum=mom)
        self.lossfn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.model(x)
        return x

    def train(self, x, correct, k):
        guess = self.forward(x)
        corr=False
        if torch.argmax(guess)==correct:
            corr=True
        loss = self.lossfn(guess, correct)
        loss.backward()
        if k%100==0 and k!=0:
            self.optim.step()
            self.optim.zero_grad()
        return loss, corr

    def save(self):
        torch.save(self.model, "layers/model.pth")
