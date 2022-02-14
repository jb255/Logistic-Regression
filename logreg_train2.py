#defense contre les force du mal
#herboristery
#ancien rhune
#astronomie

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
# %matplotlib inline
np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})

data = pd.read_csv("dataset_train.csv")
data = data.drop(columns = ['First Name', 'Last Name', 'Birthday', 'Best Hand'])
data = data.drop(columns = ['Arithmancy', 'Care of Magical Creatures', 'Astronomy'])
data = data.dropna()
data

X = data.values[:, 7:9]
X = np.column_stack((data['Herbology'] ,data['Defense Against the Dark Arts']))
y = np.column_stack(data['Hogwarts House'])
# X = np.c_[np.ones(X.shape[0]), X]

def normalise(x):
    return (x - np.mean(x)) / np.std(x)

X = normalise(X)
X = np.c_[np.ones(X.shape[0]), X]
X

def normalise(x):
    return (x - np.mean(x)) / np.std(x)

X = normalise(X)
X = np.c_[np.ones(X.shape[0]), X]
X

theta = np.zeros(3)

def sigmoid(z):
    #print (np.exp(-z))
    return (1 / (1 + np.exp(-z)))

def predict(X, theta):
    #print(X.shape, theta.shape)
    res = sigmoid(np.dot(X, theta.T))
    #print(res.shape)
    #print(res)
    return(res)

def cost(X, y, theta):
    return((-1 / X.shape[0]) * np.sum(y * np.log(predict(X, theta)) + (1 - y) * np.log(1 - predict(X, theta))))

mask_Ravenclaw = y == "Ravenclaw"
mask_Gryffindor = y == "Gryffindor"
mask_Hufflepuff = y == "Hufflepuff"
mask_Slytherin = y == "Slytherin"

theta = np.zeros(3, dtype=float)
print (cost(X, mask_Ravenclaw, theta))

def fit(X, y, theta, alpha, num_iters):
    # Initialiser certaines variables utiles
    m = X.shape[0]
    print (m)
    J_history = []
    for _ in range(num_iters):
        theta = theta - (alpha / m) * (np.dot(predict(X, theta) - y, X))
        #print(theta)
        J_history.append(cost(X, y, theta))
    return theta, J_history

theta = np.zeros(3, dtype=float)
print (X.shape, theta.shape, y.shape)
theta, J_history = fit(X, mask_Ravenclaw, theta, 0.001, 600)

cost(X, mask_Ravenclaw, theta)


fig = plt.figure()
ax = plt.axes()
ax.plot(J_history)