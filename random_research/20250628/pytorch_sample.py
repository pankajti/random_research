import torch
from torch import nn
import random

def get_sample_data(m = 4, c= 9):
    x = torch.linspace(1,10,100).unsqueeze(1)
    eps = torch.randn([100,1])
    y = m*x+c +eps
    return x,y

class SimpleNN(nn.Module):

    def __init__(self, in_features = 1, out_features = 1):
        super().__init__()  # <-- Required to initialize nn.Module properly!

        self.generator = nn.Sequential(nn.Linear(in_features=in_features,out_features=out_features),)


    def forward(self, x):
        out = self.generator(x)
        return out

import matplotlib.pyplot as plt


if __name__ == '__main__':
    x,y = get_sample_data(4,2)
    simple_nn = SimpleNN()
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(simple_nn.parameters(),lr=.01)
    losses = []

    for   i in range(1000):
        optimizer.zero_grad()
        out = simple_nn(x)
        l = loss(out, y )
        l.backward()
        optimizer.step()
        for p in simple_nn.parameters():
            print(p.data.numpy())
        losses.append(l.data.numpy())
        print(f"loss:{l}")

    with torch.no_grad():
        o = simple_nn(torch.tensor([[9.]]))
        print(o.data.numpy())
        print(l)

    plt.scatter(list(range(len(losses))), losses)
    plt.show()