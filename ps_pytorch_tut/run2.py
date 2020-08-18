import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# data = np.loadtxt('input.csv', dtype='float,float', delimiter=',')

I = 0.1418*np.ones(20)                       # kgm^2
theta = 50*(np.pi/180)

w = np.random.normal(173, 12, 20)
w = w*(np.pi/180)                            # rad/sec

t = theta/w                                  # sec
torque = (I*w)/t                             # kgm^2/s^2

w = np.expand_dims(w,axis=1)
t = np.expand_dims(t,axis=1)
torque = np.expand_dims(torque,axis=1)

input = np.append(t,torque,axis=1)

X_train = torch.FloatTensor(input)
Y_train = torch.FloatTensor(w)

#### MODEL ARCHITECTURE ####

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        hidden = nn.Sigmoid()#(self.i2h(combined))
        # self.linear = torch.nn.Linear(2,2)
        # self.lin2 = torch.nn.Linear(2,2)

    def forward(self, x):
        # x = self.lin2(x)
        y_pred = self.sigmoid(x)
        return y_pred

model = Model()

loss_func = nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
#print(len(list(model.parameters())))
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

### TRAINING
for epoch in range(2):
    y_pred = model(X_train)

    loss = loss_func(y_pred, Y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    count = count_params(model)
    print(count)

test_exp = torch.FloatTensor([[0.3]])
print("If u have 6 yrs exp, Salary is:", model(test_exp).data[0][0].item())
