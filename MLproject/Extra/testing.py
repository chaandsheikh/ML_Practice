import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
import sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
from sklearn.preprocessing import scale
from collections import Counter
from sklearn import datasets


trainingData = pd.read_csv("trainingData.csv", delimiter=',')
testData = pd.read_csv("testData.csv", delimiter=',')
XTrain=trainingData[:, :-1]
YTrain=trainingData[:, -1]
model=KNeighborsClassifier(n_neighbors=3,p=2,metric='minkowski')
model.fit(XTrain, YTrain)