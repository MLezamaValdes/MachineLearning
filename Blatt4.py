'''
Created on 22.11.2018

@author: Maite Lezama Valdes
'''

import numpy as np
import scipy.stats as sp
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math
from math import sqrt
from numpy.core.multiarray import correlate
import csv
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn import preprocessing as norm
from numpy.core.fromnumeric import std
from plotly.graph_objs import *

#### 1
path = "C:\\Users\\mleza\\Documents\\Uni\\MaschinellesLernen\\Uebung\\winequality-red.csv"
wq = pd.read_csv(filepath_or_buffer=path, sep=";")
wqc = wq[['residual sugar','density','pH','alcohol','quality']]

# b)
scatter_matrix(wqc, alpha=0.2)
# c)
fig1, ax1 = plt.subplots()
ax1.set_title('sulphates')
ax1.boxplot(wq['sulphates'].tolist())

fig2, ax1 = plt.subplots()
ax1.set_title('citric acid')
ax1.boxplot(wq['citric acid'].tolist())

#### 3
pathd = "C:\\Users\\mleza\\Documents\\Uni\\MaschinellesLernen\\Uebung\\dat.csv"
dat = pd.read_csv(filepath_or_buffer=pathd, sep=",")

df = dat[['x1', 'x2']]

# skalieren
xstd = norm.StandardScaler().fit_transform(df)

# Kovarianzmatrix
mv = np.mean(xstd, axis=0)
comat = (xstd - mv).T.dot((xstd - mv)) / (xstd.shape[0]-1)

# Eigenwert und Eigenvektor der Kovarianzmatrix
eig_vals, eig_vecs = np.linalg.eig(comat)

# tupels aus Ewert und Evektor
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Eigenwerte 2.152 und 0.069
    
tot_val = sum(eig_vals)
var_exp = [(i / tot_val)*100 for i in sorted(eig_vals, reverse=True)]
# PC 1 explains 96.86 % of variance, PC 2 the remaining 3.14 % 

# Eigenvektormatrix 
matrix_w = np.hstack((eig_pairs[0][1].reshape(2,1), 
                      eig_pairs[1][1].reshape(2,1)))

# project to new feature space -> Scores
scores = pd.DataFrame(xstd.dot(matrix_w))
scores.columns = ['PC1', 'PC2']
pc1 = scores['PC1'].tolist()
pc2 = scores['PC2'].tolist()

# plot results
plt.scatter(pc1, pc2)
plt.ylabel("PC2")
plt.xlabel("PC1")
plt.title("scores PCA")

