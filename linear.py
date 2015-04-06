import csv
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

fileName  = "train.csv"
data = getData(fileName)

hour = propertyData('hour')
month = propertyData('month')
season = propertyData('season')
weather = propertyData('weather')
workingDay = propertyData('workingday')
holiday = propertyData('holiday')
humidity = propertyData('humidity')
temp = propertyData('temp')
year = propertyData('year')

result = data['count']

tempData = []

tempData.append(assignWeights(hour,data))
tempData.append(assignWeights(month,data))
tempData.append(assignWeights(season,data))
tempData.append(assignWeights(weather,data))
tempData.append(assignWeights(workingDay,data))
tempData.append(assignWeights(holiday,data))
#tempData.append(assignWeights(humidity,data))
#tempData.append(assignWeights(temp,data))
#tempData.append(assignWeights(year,data))
tempData.append(data['temp'])
tempData.append(data['year'])
newData = []

for i in range(len(tempData[0])):
    slice = []
    for j in range(len(tempData)):
        slice.append(tempData[j][i])
    newData.append(slice)

del data['count']
del data['registered']
del data['casual']

del data['atemp']
del data['windspeed']
#del data['season']
#del data['weather']
#del data['humidity']
del data['weekday']
#del data['holiday']
#del data['workingday']
#del data['month']
#del data['hour']
#del data['temp']
#del data['year']

regr = linear_model.LinearRegression()
regr.fit(newData,result)
calc = regr.predict(newData)

for i,x in enumerate(calc):
    if x < 0:
        calc[i] = 0
for i,res in enumerate(calc):
    if data['hour'][i] in range(0,7):
        calc[i] = 0

r = 10000
l = 0
u = l + r

x = range(r)

plt.scatter(x, result[l:u], color = 'red')
plt.plot(x, calc[l:u], color = 'blue')
plt.show()

print mean_squared_error(result[l:u],calc[l:u])
