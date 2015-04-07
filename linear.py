import csv
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from datetime import datetime, date, time
from sklearn.feature_selection import VarianceThreshold
from sklearn import linear_model 
from dataInput import *

class propertyData:
    def __init__(self, name):
        self.name = name
        self.sum = 0
        self.dict = {}

# split into 2 for learning and replacing!
def assignWeights(obj,data):
    for i,value in enumerate(data[obj.name]):
        count = result[i]
        obj.sum += count
        
        if value in obj.dict:
            obj.dict[value] += count
        else:
            obj.dict[value] = count

    for value in obj.dict:
        obj.dict[value] /= float(obj.sum)
    newData = []
    for value in data[obj.name]:
         newData.append(obj.dict[value])

    return newData

def replaceTestValues(testData, obj):
    for i,x in enumerate(testData[obj.name]):
        testData[obj.name][i] = obj.dict[x]

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

hour = propertyData('hour')
month = propertyData('month')
season = propertyData('season')
weather = propertyData('weather')
workingDay = propertyData('workingday')
holiday = propertyData('holiday')
humidity = propertyData('humidity')
temp = propertyData('temp')
year = propertyData('year')


trainData['hour'] = assignWeights(hour,trainData)
trainData['month'] = assignWeights(month,trainData)
trainData['holiday'] = assignWeights(holiday,trainData)
replaceTestValues(testData,hour)
replaceTestValues(testData,month)
replaceTestValues(testData,holiday)

"""
tempData.append(trainData['temp'])
tempData.append(trainData['year'])
tempData.append(trainData['workingday'])
tempData.append(trainData['humidity'])
tempData.append(trainData['windspeed'])
tempData.append(trainData['weekday'])
tempData.append(trainData['season'])
tempData.append(trainData['weather'])
newData = []
"""

"""
for i in range(len(tempData[0])):
    slice = []
    for j in range(len(tempData)):
        slice.append(tempData[j][i])
    newData.append(slice)
"""

poly = PolynomialFeatures(degree = 4)
polyDataTrain = poly.fit_transform(trainData)
polyDataTest = poly.fit_transform(testData)

regr = linear_model.LinearRegression()
regr.fit(polyDataTrain,result)

calc1 = regr.predict(polyDataTrain)
calc = regr.predict(polyDataTest)

for i,x in enumerate(calc1):
    if x < 0:
        calc1[i] = 0

for i,x in enumerate(calc):
    if x < 0:
        calc[i] = 0

with open('output_poly.csv', 'w') as csvfile:
    fieldNames = ['datetime', 'count']
    writer = csv.DictWriter(csvfile,fieldnames=fieldNames)
    writer.writeheader()
    
    for i in range(len(dateTime)):
        writer.writerow({'datetime':dateTime[i], 'count':calc[i]})

"""
for i,res in enumerate(calc):
    if data['hour'][i] in range(0,7):
        calc[i] = 0
"""

r = 10000
l = 0 
u = l + r

x = range(r)

plt.scatter(x, result[l:u], color = 'red')
plt.plot(x, calc1[l:u], color = 'blue')
plt.show()

print mean_squared_error(result[l:u],calc1[l:u])
