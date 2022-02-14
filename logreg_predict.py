import pandas as pd
import numpy as np
import math

thetas = np.loadtxt('thetas.csv', delimiter=',')

thetaG = thetas[0]
thetaS = thetas[1]
thetaH = thetas[2]
thetaR = thetas[3]

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

def predict(X, theta):
    res = sigmoid(np.dot(X, theta.T))
    return(res)

test = pd.read_csv("dataset_test.csv")

ids = test['Index']

test = test.drop(columns = ['Hogwarts House'])

test = test.drop(columns = ['First Name', 'Last Name', 'Birthday', 'Best Hand', 'Index'])
test = test.drop(columns = ['Arithmancy', 'Care of Magical Creatures', 'Flying', 'Potions', 'Charms', 'Transfiguration'])#, 'Astronomy'])
# 'Divination', 'Muggle Studies', 'History of Magic', 'Transfiguration', 'Potions', 'Flying'])

col_mean = np.nanstd(test['Herbology'], axis=0)
test['Herbology'][np.isnan(test['Herbology'])] = col_mean

col_mean = np.nanstd(test['Defense Against the Dark Arts'], axis=0)
test['Defense Against the Dark Arts'][np.isnan(test['Defense Against the Dark Arts'])] = col_mean

col_mean = np.nanstd(test['Divination'], axis=0)
test['Divination'][np.isnan(test['Divination'])] = col_mean

col_mean = np.nanstd(test['Muggle Studies'], axis=0)
test['Muggle Studies'][np.isnan(test['Muggle Studies'])] = col_mean

col_mean = np.nanstd(test['History of Magic'], axis=0)
test['History of Magic'][np.isnan(test['History of Magic'])] = col_mean

# col_mean = np.nanstd(test['Transfiguration'], axis=0)
# test['Transfiguration'][np.isnan(test['Transfiguration'])] = col_mean

# col_mean = np.nanstd(test['Potions'], axis=0)
# test['Potions'][np.isnan(test['Potions'])] = col_mean

# col_mean = np.nanstd(test['Flying'], axis=0)
# test['Flying'][np.isnan(test['Flying'])] = col_mean

col_mean = np.nanstd(test['Ancient Runes'], axis=0)
test['Ancient Runes'][np.isnan(test['Ancient Runes'])] = col_mean

# col_mean = np.nanstd(test['Charms'], axis=0)
# test['Charms'][np.isnan(test['Charms'])] = col_mean

col_mean = np.nanstd(test['Astronomy'], axis=0)
test['Astronomy'][np.isnan(test['Astronomy'])] = col_mean



def normalise(x):
    return (x - np.mean(x)) / np.std(x)

test = normalise(test)


# a = 0
# while a < 11:
# 	col_mean = np.nanstd(test[a], axis=0)
# 	test[a][np.isnan(test[a])] = col_mean
#	a += 1

# test = np.column_stack((test['Herbology'] ,test['Defense Against the Dark Arts']))

test = np.c_[np.ones(test.shape[0]), test]


R = predict(test, thetaR)
G = predict(test, thetaG)
S = predict(test, thetaS)
H = predict(test, thetaH)



result = np.array([["Index", "Hogwarts House"]])
for i in range(len(R)):
	is_r = True if R[i] > G[i] and R[i] > S[i] and R[i] > H[i] else False
	is_g = True if G[i] > R[i] and G[i] > S[i] and G[i] > H[i] else False
	is_s = True if S[i] > R[i] and S[i] > G[i] and S[i] > H[i] else False
	is_h = True if H[i] > R[i] and H[i] > G[i] and H[i] > S[i] else False

	if is_r :
		house = "Ravenclaw"
	elif is_g :
		house = "Gryffindor"
	elif is_s :
		house = "Slytherin"
	elif is_h:
		house = "Hufflepuff"
	else:
		house = "??"
	result = np.append (result, [[ids[i], house]], axis = 0)

np.savetxt('houses.csv', result, delimiter=',', fmt="%s")