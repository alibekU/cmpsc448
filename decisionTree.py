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


trainFile  = "train.csv"
testFile = "test.csv"

trainData = getData(trainFile)
testData = getData(testFile)

dateTime = testData['datetime']

result = trainData['count']

del trainData['datetime']
del trainData['count']
del trainData['registered']
del trainData['casual']
del testData['datetime']

regr = tree.DecisionTreeClassifier()
regr.fit(trainData,result)
calc = regr.predict(testData)

for i,x in enumerate(calc):
    if x < 0:
        calc[i] = 0

with open('output.csv', 'w') as csvfile:
    fieldNames = ['datetime', 'count']
    writer = csv.DictWriter(csvfile,fieldnames=fieldNames)
    writer.writeheader()
    
    for i in range(len(dateTime)):
        writer.writerow({'datetime':dateTime[i], 'count':calc[i]})
"""
r = 10000
l = 0 
u = l + r

x = range(r)

plt.scatter(x, result[l:u], color = 'red')
plt.plot(x, calc[l:u], color = 'blue')
plt.show()

print mean_squared_error(result[l:u],calc[l:u])
"""
