import csv
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from datetime import datetime, date, time
from sklearn.feature_selection import VarianceThreshold
from sklearn import linear_model 

def getData(fileName,):
    dataIn = []

    header = []

    with open(fileName, "r") as f:
            header = f.next().strip("\n").split(",")
            for line in f:
                dataIn.append(line.strip("\n").split(","))

    data  = pd.DataFrame(data=np.asarray(dataIn), columns=header)

    data['hour'] = data['datetime'].map(lambda x: (datetime.strptime(x, "%Y-%m-%d %H:%M:%S")).hour)
    data['weekday'] = data['datetime'].map(lambda x: (datetime.strptime(x, "%Y-%m-%d %H:%M:%S")).weekday())
    data['month'] = data['datetime'].map(lambda x: (datetime.strptime(x, "%Y-%m-%d %H:%M:%S")).month)
    data['year'] = data['datetime'].map(lambda x: (datetime.strptime(x, "%Y-%m-%d %H:%M:%S")).year)

    for feature in data:
        if feature != 'datetime':
            data[feature] = map(float, data[feature])
    
    return data
