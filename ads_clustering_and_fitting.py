# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 02:16:06 2023

@author: HP
"""

import pandas as pds
import numpy as npy
from sklearn import cluster
import matplotlib.pyplot as pylt
from scipy.optimize import curve_fit
import sklearn.metrics as skmet

"""
CLUSTERING
"""

prob = pds.read_csv('probability_of_dying_youth_20_24.csv', skiprows=(3))
print(prob.head())


prob = prob.drop(['Unnamed: 66','Country Code', 'Indicator Name', 'Indicator Code', '1960',
                '1961','1962','1963','1964','1965','1966','1967','1968',
                '1969','1970','1971','1972','1973','1974','1975','1976',
                '1977','1978','1979','1980','1981','1982','1983','1984',
                '1985','1986','1987','1988','1989'], axis=1)
print(prob.head())

prob = prob.fillna(0)
print("\nNew Probability of dying among youth ages 20-24 years(per 1000) after filling null values: \n", prob.head())

prob = pds.DataFrame.transpose(prob)
print("\nTransposed Dataframe: \n",prob.head())

header = prob.iloc[0].values.tolist()
prob.columns = header
print("\nProbability Header: \n",prob)

prob= prob.iloc[2:]
print("\nNew Transposed Dataframe: \n",prob)

prob_ex = prob[["India","United Kingdom"]].copy()

max_val = prob_ex.max()
min_val = prob_ex.min()
prob_ex = (prob_ex - min_val) / (max_val - min_val)
print("\nNew selected columns dataframe: \n", prob_ex)

ncluster = 5
kmeans = cluster.KMeans(n_clusters=ncluster)

kmeans.fit(prob_ex)

labels = kmeans.labels_

# extract the estimated cluster centres
cen = kmeans.cluster_centers_
print(cen)

# calculate the silhoutte score
print(skmet.silhouette_score(prob_ex, labels))

# plot using the labels to select colour
pylt.figure(figsize=(10.0, 10.0))
col = ["tab:purple", "tab:pink", "tab:orange", "tab:brown", "tab:green", "tab:red", \
"tab:yellow", "tab:gray", "tab:cyan", "tab:blue"]
    
for l in range(ncluster): # loop over the different labels
    pylt.plot(prob_ex[labels==l]["India"], prob_ex[labels==l]["United Kingdom"], marker="o", markersize=3, color=col[l])    
    
# show cluster centres
for ic in range(ncluster):
    xc, yc = cen[ic,:]  
    pylt.plot(xc, yc, "dk", markersize=10)
pylt.xlabel("India")
pylt.ylabel("United Kingdom")
pylt.title("Probability of dying youth in United Kingdom v/s India by cluster")
pylt.legend()
pylt.show()    

print(cen)

df_cen = pds.DataFrame(cen, columns=["India", "United Kingdom"])
print(df_cen)
df_cen = df_cen * (max_val - min_val) + max_val
prob_ex = prob_ex * (max_val - min_val) + max_val
# print(df_ex.min(), df_ex.max())

print(df_cen)

# plot using the labels to select colour
pylt.figure(figsize=(10.0, 10.0))

col = ["tab:purple", "tab:pink", "tab:orange", "tab:brown", "tab:green", "tab:red", \
"tab:yellow", "tab:gray", "tab:cyan", "tab:blue"]
for l in range(ncluster): # loop over the different labels
    pylt.plot(prob_ex[labels==l]["India"], prob_ex[labels==l]["United Kingdom"], "o", markersize=3, color=col[l])
    
# show cluster centres

pylt.plot(df_cen["India"], df_cen["United Kingdom"], "dk", markersize=10)
pylt.xlabel("India")
pylt.ylabel("United Kingdom")
pylt.title("Probability of dying youth in United Kingdom v/s India by cluster")
pylt.legend()
pylt.show()
print(cen)    

"""
CURVE FIT 
"""

Mort = pds.read_csv("Mortality_rate_under_5.csv", skiprows=(3))
print("\nMortality rate under 5: \n", Mort.head())

Mort = pds.DataFrame(Mort)


Mort = Mort.transpose()
Mort = Mort.fillna(0)
print("\nMortality rate under 5: \n", Mort.head())


header3 = Mort.iloc[0].values.tolist()
Mort.columns = header3
print("\nMortality rate Header: \n",Mort.head())

Mort = Mort["United Arab Emirates"]
print("\nMortality rate after dropping columns: \n", Mort)


Mort.columns = ["Mortality rate"]
print("\nMortality rate: \n",Mort)


Mort = Mort.iloc[5:]
Mort = Mort.iloc[:-1]
print("\nMortality rate: \n",Mort)


Mort = Mort.reset_index()
print("\nMortality rate index: \n",Mort)


Mort = Mort.rename(columns={"index": "Year", "United Arab Emirates": "Mortality rate"} )
print("\nMortality rate rename: \n",Mort)


print(Mort.columns)
Mort.plot("Year", "Mortality rate")
pylt.title("Mortality Rate in Years")
pylt.show()


def exponential(s, q0, h):
    #Calculates exponential function with scale factor q0 and Mortality rate
    s = s - 1982.0
    x = q0 * npy.exp(h*s)
    return x

print(type(Mort["Year"].iloc[1]))
Mort["Year"] = pds.to_numeric(Mort["Year"])
print("\nMortality Type: \n", type(Mort["Year"].iloc[1]))
paramt, covar = curve_fit(exponential, Mort["Year"], Mort["Mortality rate"],
p0=(4.978423, 0.03))


Mort["fit"] = exponential(Mort["Year"], *paramt)
Mort.plot("Year", ["Mortality rate", "fit"], label=["Mortality rate", "Fit"])
pylt.title("Mortality rate and fitting of country United Arab Emirates")
pylt.legend()
pylt.show()


def err_ranges(x, exponential, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    """
    import itertools as iter
    
    # initiates arrays for lower and upper limits
    lower = exponential(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = exponential(x, *p)
        lower = npy.minimum(lower, y)
        upper = npy.maximum(upper, y)
        print("\nLower: \n", lower)
        print("\nUpper: \n", upper)        
    return lower, upper



