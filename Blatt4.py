# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
'''
Created on 22.11.2018

@author: mleza
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

#### 1
#%%
path = "C:\\Users\\mleza\\Documents\\Uni\\MaschinellesLernen\\Uebung\\winequality-red.csv"
wq = pd.read_csv(filepath_or_buffer=path, sep=";")
list(wq.columns.values)
wqc = wq[['residual sugar','density','pH','alcohol','quality']]
wqc
sns.pairplot(wq)
#%%

# a
# Scatterplot Matrix auf den Daten 'residual sugar', 'density', 'pH', alcohol' und 
# 'quality'. Gibt es Korrelationen? 
scatter_matrix(wq, alpha=0.2)
plt.show()

sns.set(style="ticks")
df = sns.load_dataset("iris")
sns.pairplot(wq, hue="quality")
