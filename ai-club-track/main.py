import numpy as np
import math
import pandas as pd


# Create random input and output data
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)
np.dataset= pd.read_csv("HousingDataset.csv")
data = np.dataset.to_numpy()
weights = [];

i =0
for i in range(12):
    weights.append(np.random.randn())

print(weights)
# Randomly initialize weights
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()






learning_rate = 1e-6
t = 1
for t in range(len(data)):
    # Forward pass: compute predicted y
    # y = a + b x + c x^2 + d x^3
    #y_pred = a + b * x + c * x ** 2 + d * x ** 3
    #print out each row
    print(data[t])
    j=2
    ypred = 0;
    for j in range 13:
        if j>=2 and j<=5 or j==11:
            ypred = ypred + weights[j-2]*data[t][j]
        else if J>=6 and j

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')