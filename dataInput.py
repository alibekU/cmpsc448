import csv
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from datetime import datetime, date, time
from sklearn.feature_selection import VarianceThreshold
from sklearn import linear_model 

class propertyData:
    def __init__(self, name):
        self.name = name
        self.sum = 0
        self.dict = {}

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

dataIn = []

fileName  = "train.csv"
header = []


hour = propertyData('hour')
month = propertyData('month')
season = propertyData('season')
weather = propertyData('weather')
workingDay = propertyData('workingday')
holiday = propertyData('holiday')
humidity = propertyData('humidity')
temp = propertyData('temp')

with open(fileName, "r") as f:
        header = f.next().strip("\n").split(",")
        for line in f:
            dataIn.append(line.strip("\n").split(","))

data  = pd.DataFrame(data=np.asarray(dataIn), columns=header)

data['hour'] = data['datetime'].map(lambda x: (datetime.strptime(x, "%Y-%m-%d %H:%M:%S")).hour)
data['weekday'] = data['datetime'].map(lambda x: (datetime.strptime(x, "%Y-%m-%d %H:%M:%S")).weekday())
data['month'] = data['datetime'].map(lambda x: (datetime.strptime(x, "%Y-%m-%d %H:%M:%S")).month)

del data['datetime']

data['temp'] = map(lambda x:float(x),data['temp'])
data['atemp'] = map(float,data['atemp'])
data['humidity'] = map(float,data['humidity'])
data['windspeed'] = map(float,data['windspeed'])
data['casual'] = map(int,data['casual'])
data['registered'] = map(int, data['registered'])
data['count'] = map(int, data['count'])
data['holiday'] = map(int, data['holiday'])
data['workingday'] = map(int, data['workingday'])
data['weather'] = map(int, data['weather'])
data['season']  = map(int, data['season'])

result = data['count']

tempData = []

tempData.append(assignWeights(hour,data))
tempData.append(assignWeights(month,data))
tempData.append(assignWeights(season,data))
tempData.append(assignWeights(weather,data))
tempData.append(assignWeights(workingDay,data))
tempData.append(assignWeights(holiday,data))
tempData.append(assignWeights(humidity,data))
tempData.append(assignWeights(temp,data))
newData = []

for i in range(len(result)):
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

regr = linear_model.LinearRegression()
regr.fit(newData, result)
calc = regr.predict(newData)

for i,x in enumerate(calc):
    if x < 0:
        calc[i] = 0

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
plt.plot(x, calc[0:r], color = 'blue')
plt.show()

print mean_squared_error(result[l:u],calc[l:u])
