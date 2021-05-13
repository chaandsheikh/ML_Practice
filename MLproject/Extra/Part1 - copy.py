import numpy as np
from sklearn.preprocessing import StandardScaler

from collections import Counter

class MLPart1:
    def __init__(self, __trainingDataFile, __testDataFile, __k):
        self.weight = 10000
        self.trainingDataFileName = __trainingDataFile
        self.testDataFileName = __testDataFile
        self.trainingData = np.array([])
        self.testData = np.array([])
        self.euclideanDistances = np.array([])
        self.euclideanDistancesSorted = np.array([])
        self.predictions = np.array([])
        self.originalCategories = np.array([])
        self.readInstances()
        self.k = __k

    def readInstances(self):
        """
        Reading an instance from fName
        """
        #sc = StandardScaler()
        self.trainingData = np.genfromtxt(self.trainingDataFileName, delimiter=',', dtype=float)
        #self.trainingData = sc.fit_transform(self.trainingData)
        self.testData = np.genfromtxt(self.testDataFileName, delimiter=',', dtype=float)
        #self.testData = sc.fit_transform(self.testData)
        self.originalCategories = self.testData[:, -1]

    # Works on local variables and returns 2 numpy array
    def calculateDistances(self,trainingInstancesMatrix, singlQueryPoint):
        data = np.sqrt(np.sum((trainingInstancesMatrix - singlQueryPoint) ** 2, axis=1))
        return data, np.argsort(data)

    # Works on global variables
    def updateDistances(self):
         distances = []
         sortedDistances = []
         for i in range(0, np.shape(self.testData)[0]):
            d, sd = self.calculateDistances(self.trainingData[:, :-1], self.testData[i, :-1])
            distances.append(d)
            sortedDistances.append(sd)
            i+= 1
         self.euclideanDistances = np.array(distances)
         self.euclideanDistancesSorted = np.array(sortedDistances)

    def predictCategories(self):
        prediction = np.array([])
        # To order tp improve performance, avoid calculation with K=1
        if self.k == 1:
            for i in range(0, len(self.euclideanDistances)):
                index = self.euclideanDistancesSorted[i][0]
                prediction = np.append(prediction, self.trainingData[index][-1])
        else:
            for i in range(0, len(self.euclideanDistances)):
                indices = self.euclideanDistancesSorted[i, :self.k]
                indice_category_distance = {}
                tie = []
                for indice in indices:
                    tie.append(self.trainingData[indice][-1])
                data = Counter(tie)
                #print(data.most_common(1)[0][0])  # Returns the highest occurring item
                prediction = np.append(prediction, data.most_common(1)[0][0])
        self.predictions = prediction


    def predictionAccuracy(self):
        instancesCount = len(self.originalCategories)
        correctPredictionCounter = 0
        for i in range(0,instancesCount):
            if self.originalCategories[i] == self.predictions[i]:
                correctPredictionCounter +=1
        print("Prediction accuracy: ", (correctPredictionCounter/instancesCount) * 100)

    def viewData(self):
        print(self.testData)
        print("*************************")
        sc = StandardScaler()
        self.testData = sc.fit_transform(self.testData)
        print(self.testData)

    def search(self):
        self.updateDistances()
        self.predictCategories()
        self.predictionAccuracy()


ml = MLPart1("trainingData.csv", "testData.csv", 1)
ml.viewData()
'''for i in range(1,11):
    ml = MLPart1("trainingData.csv", "testData.csv", i)
    ml.search()'''


