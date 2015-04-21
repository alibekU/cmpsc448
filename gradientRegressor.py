import csv
from sklearn import tree 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from datetime import datetime, date, time
from sklearn.feature_selection import VarianceThreshold
from sklearn import linear_model 
from dataInput import *
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

trainFile  = "train.csv"
testFile = "test.csv"

trainData = getData(trainFile)
testData = getData(testFile)

dateTime = testData['datetime']

reg = trainData['registered']
cas = trainData['casual']
result = trainData['count']

del trainData['datetime']
del trainData['count']
del trainData['registered']
del trainData['casual']


#del trainData['season']
#del trainData['holiday']
#del trainData['workingday']
del trainData['atemp']
#del trainData['humidity']
#del trainData['windspeed']
#del trainData['weekday']
#del trainData['month']
#del trainData['year']


del testData['datetime']

#del testData['season']
#del testData['holiday']
#del testData['workingday']
del testData['atemp']
#del testData['humidity']
#del testData['windspeed']
#del testData['weekday']
#del testData['month']
#del testData['year']

regrReg = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0, loss = 'ls')
regrCas = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0, loss = 'ls')
regrReg.fit(trainData,reg)
regrCas.fit(trainData,cas)

calcReg = regrReg.predict(testData)
calcCas = regrCas.predict(testData)
calcReg1 = regrReg.predict(trainData)
calcCas1 = regrCas.predict(trainData)

for i,x in enumerate(calcReg):
    if x < 0:
        calcReg[i] = 0

for i,x in enumerate(calcReg1):
    if x < 0:
        calcReg1[i] = 0

for i,x in enumerate(calcCas):
    if x < 0:
        calcCas[i] = 0

for i,x in enumerate(calcCas1):
    if x < 0:
        calcCas1[i] = 0

calc = [calcReg[i] + calcCas[i] for i in range(len(calcReg))]
calc1 = [calcReg1[i] + calcCas1[i] for i in range(len(calcReg1))]

with open('output.csv', 'w') as csvfile:
    fieldNames = ['datetime', 'count']
    writer = csv.DictWriter(csvfile,fieldnames=fieldNames)
    writer.writeheader()
    
    for i in range(len(dateTime)):
        writer.writerow({'datetime':dateTime[i], 'count':calc[i]})

r = 10000
l = 0 
u = l + r

x = range(r)

plt.scatter(x, result[l:u], color = 'red')
plt.plot(x, calc1[l:u], color = 'blue')
plt.show()

print mean_squared_error(result[l:u],calc1[l:u])
