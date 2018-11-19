'''
Created on 13.11.2018

@author: Maite Lezama Valdes
'''
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
import pandas as pd
import math
from math import sqrt
from numpy.core.multiarray import correlate
import csv


########### 1
x = [4,2,5,6,1,6,8,3,4,9]

#a) SD
print(np.std(x))

#b) Schiefe
plt.hist(x)
sp.skew(x)

#c) Quartile 
np.percentile(x, [25, 50, 75])

#d) Ausreiﬂer
plt.boxplot(x)

def testotl(x):
    mx = np.mean(x)
    sdx = np.std(x)
    ra = [mx-2*sdx, mx+2*sdx]
    res = []
    for i in range(len(x)):
        if ra[0] <= x[i] <= ra[1]:
            res.append(0)
        else:
            res.append(1)
    rest = "no outliers in data is:", all(res == 0 for res in res)
    return(res, rest)
#end 
testotl(x)

########### 2
#a)
x = [3,4,5,345,14,235,57,43]
y = [34,34,657,23,645,8,312,345]

def mymean(x):
    return float(sum(x)) / max(len(x), 1)
def corfunc(x, y):
    xm = mymean(x)
    ym = mymean(y)
    cov = []
    xs = []
    ys = []
    for i in range(len(x)): 
        cov.append((x[i]-xm)*(y[i]-ym))
        xs.append((x[i]-xm)**2)
        ys.append((y[i]-ym)**2)
    r = sum(cov)/math.sqrt(sum(xs)*sum(ys))
    return(r)

#b)
path = "C:\\Users\\mleza\\Documents\\Uni\\MaschinellesLernen\\Uebung\\icecream.csv"
ice = pd.read_csv(filepath_or_buffer=path)
# check for outliers with 1.5-times IQR and 2xSD
plt.boxplot(ice.Ice_Cream_Sales)
testotl(ice.Ice_Cream_Sales)
testotl(ice.Temperature)
#no outliers detected
#check weather there are any incomplete cases
pd.DataFrame.count(ice.dropna()) == pd.DataFrame.count(ice)
# no rows contain missing values

#c) 
corfunc(ice.Temperature, ice.Ice_Cream_Sales)

#d)
# very differing value ranges, thus normalization
ice_norm = (ice - ice.mean()) / (ice.max() - ice.min())
plt.plot(ice_norm.index, 'Ice_Cream_Sales', data=ice_norm, marker='', color='red', linewidth=2, label='Ice Cream Sales')
plt.plot(ice_norm.index, 'Temperature', data=ice_norm, marker='', color='blue', linewidth=2, label="Temperature")
plt.ylabel("normalized Temperature (blue) and Ice Cream Sales (red)")
plt.xlabel("sample")


