import numpy as np
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

np.random.seed(1)

I = 0.1418*np.ones(20)                       # kgm^2
theta = 50*(np.pi/180)

w = np.random.normal(173, 12, 20)
w = w*(np.pi/180)                            # rad/sec

t = theta/w                                  # sec
torque = (I*w)/t                             # kgm^2/s^2

w = np.expand_dims(w,axis=1)
t = np.expand_dims(t,axis=1)
torque = np.expand_dims(torque,axis=1)

# Input (time, torque)
inputs = np.append(t,torque,axis=1)

# Output (ang velocity)
outputs = w

inputs = torch.from_numpy(inputs)
outputs = torch.from_numpy(outputs)

# Weights and bias
w = torch.randn((1,2), requires_grad=True)
b = torch.randn((1), requires_grad=True)

# Model
def model(x):
    return x @ w.t() + b

# Predictions
# pred = model(inputs.float())
# print(pred)
# print
# print(outputs)

# Error (MSE)
def mse(t1, t2):
    diff = t1-t2
    return torch.sum(diff**2)/diff.numel()

# # Loss
# loss = mse(pred,outputs)
#
# # Gradient
# loss.backward()
#
# print(w)
# print(w.grad)
#
# w = w - w.grad*1e-5
# b = b - b.grad*1e-5
#
# w.grad.zero()
# b.grad.zero()

# Loss
# pred = model(inputs.float())
# loss = mse(pred, outputs)

for i in range(10):
    pred = model(inputs.float())
    loss = mse(pred, outputs)
    loss = loss.mean()
    loss.backward()
    with torch.no_grad():
        w = w - w.grad*1e-5
        b = b - b.grad*1e-5

        # w.grad = 0
        w.grad.zero_()
        b.grad.z

# Loss
pred = model(inputs)
loss = mse(pred, outputs)

print(loss)
