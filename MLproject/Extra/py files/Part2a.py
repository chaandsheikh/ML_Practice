import numpy as np
import statistics as stat

class MLPart2a:
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
        self.trainingData = np.genfromtxt(self.trainingDataFileName, delimiter=',', dtype=float)
        self.testData = np.genfromtxt(self.testDataFileName, delimiter=',', dtype=float)
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
        # To order to improve performance, avoid calculation with K=1
        if self.k == 1:
            for i in range(0, len(self.euclideanDistances)):
                index = self.euclideanDistancesSorted[i][0]
                prediction = np.append(prediction, self.trainingData[index][-1])
        else:
            for i in range(0, len(self.euclideanDistances)):
                indices = self.euclideanDistancesSorted[i, :self.k]
                indice_category_distance = {}
                for indice in indices:
                    indice_category_distance[indice] = [self.trainingData[indice][-1], self.euclideanDistances[i][indice]]
                cat_0_values = np.array([])
                cat_1_values = np.array([])
                cat_2_values = np.array([])

                for indice_values in indice_category_distance.values():
                    if int(indice_values[0]) == 0:
                        cat_0_values = np.append(cat_0_values, 1/indice_values[1])
                    elif int(indice_values[0]) == 1:
                        cat_1_values = np.append(cat_1_values, 1/indice_values[1])
                    else:
                        cat_2_values = np.append(cat_2_values, 1/indice_values[1])
                cat0 = 0
                cat1 = 0
                cat2 = 0
                if len(cat_0_values) != 0:
                    cat0 = np.sum(cat_0_values)
                if len(cat_1_values) != 0:
                    cat1 = np.sum(cat_1_values)
                if len(cat_2_values) != 0:
                    cat2 = np.sum(cat_2_values)

                cat = np.where([cat0, cat1, cat2] == np.max([cat0, cat1, cat2]))

                if cat[0][0] == 0:
                    prediction = np.append(prediction, 0.0)
                elif cat[0][0] == 1:
                    prediction = np.append(prediction, 1.0)
                else:
                    prediction = np.append(prediction, 2.0)
        self.predictions = prediction

    def predictionAccuracy(self):
        instancesCount = len(self.originalCategories)
        correctPredictionCounter = 0
        for i in range(0,instancesCount):
            if int(self.originalCategories[i]) == int(self.predictions[i]):
                correctPredictionCounter += 1
        return (correctPredictionCounter/instancesCount) * 100

    def search(self):
        self.updateDistances()
        self.predictCategories()
        return self.predictionAccuracy()


ml = MLPart2a("data\\classification\\trainingData.csv", "data\\classification\\testData.csv", 1)
print("Prediction accuracy: ", ml.search())


