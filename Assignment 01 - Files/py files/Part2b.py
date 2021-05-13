import numpy as np
from collections import Counter

class MLPart2a:
    def __init__(self, __trainingDataFile, __testDataFile, __k, __distance_weighted=True, __distance_measurements_method= "Euclidean", __weight_power = 1):
        self.weight = 10000
        self.trainingDataFileName = __trainingDataFile
        self.testDataFileName = __testDataFile
        self.trainingData = np.array([])
        self.testData = np.array([])
        self.distances = np.array([])
        self.distancesSorted = np.array([])
        self.predictions = np.array([])
        self.originalCategories = np.array([])
        self.readInstances()
        self.distance_weighted = __distance_weighted
        self.k = __k
        self.distance_measurements_method = __distance_measurements_method.lower()
        self.weight_power = __weight_power

    def readInstances(self):
        """
        Reading an instance from fName
        """
        self.trainingData = np.genfromtxt(self.trainingDataFileName, delimiter=',', dtype=float)
        self.testData = np.genfromtxt(self.testDataFileName, delimiter=',', dtype=float)
        self.originalCategories = self.testData[:, -1]

    # Works on local variables and returns 2 numpy array
    def calculateDistancesEuclidean(self,trainingInstancesMatrix, singlQueryPoint):
        data = np.sqrt(np.sum((trainingInstancesMatrix - singlQueryPoint) ** 2, axis=1))
        return data, np.argsort(data)

    # Works on global variables
    def updateDistancesEuclidean(self):
         distances = []
         sortedDistances = []
         for i in range(0, np.shape(self.testData)[0]):
            d, sd = self.calculateDistancesEuclidean(self.trainingData[:, :-1], self.testData[i, :-1])
            distances.append(d)
            sortedDistances.append(sd)
            i+= 1
         self.distances = np.array(distances)
         self.distancesSorted = np.array(sortedDistances)

         # Works on local variables and returns 2 numpy array

    def calculateDistancesManhattan(self, trainingInstancesMatrix, singlQueryPoint):
        data = np.sum(np.absolute(trainingInstancesMatrix - singlQueryPoint), axis=1)
        return data, np.argsort(data)

        # Works on global variables

    def updateDistancesManhattan(self):
        distances = []
        sortedDistances = []
        for i in range(0, np.shape(self.testData)[0]):
            d, sd = self.calculateDistancesManhattan(self.trainingData[:, :-1], self.testData[i, :-1])
            distances.append(d)
            sortedDistances.append(sd)
            i += 1
        self.distances = np.array(distances)
        self.distancesSorted = np.array(sortedDistances)
    def predictCategoriesBasic(self):
        prediction = np.array([])
        # To order tp improve performance, avoid calculation with K=1
        if self.k == 1:
            for i in range(0, len(self.distances)):
                index = self.distancesSorted[i][0]
                prediction = np.append(prediction, self.trainingData[index][-1])
        else:
            for i in range(0, len(self.distances)):
                indices = self.distancesSorted[i, :self.k]
                tie = []
                for indice in indices:
                    tie.append(self.trainingData[indice][-1])
                data = Counter(tie)
                #print(data.most_common(1)[0][0])  # Returns the highest occurring item
                prediction = np.append(prediction, data.most_common(1)[0][0])
        self.predictions = prediction


    def predictCategoriesWeighted(self):
        prediction = np.array([])
        # To order to improve performance, avoid calculation with K=1
        if self.k == 1:
            for i in range(0, len(self.distances)):
                index = self.distancesSorted[i][0]
                prediction = np.append(prediction, self.trainingData[index][-1])
        else:
            for i in range(0, len(self.distances)):
                indices = self.distancesSorted[i, :self.k]
                indice_category_distance = {}
                for indice in indices:
                    indice_category_distance[indice] = [self.trainingData[indice][-1], self.distances[i][indice]]
                cat_0_values = np.array([])
                cat_1_values = np.array([])
                cat_2_values = np.array([])

                for indice_values in indice_category_distance.values():
                    if int(indice_values[0]) == 0:
                        cat_0_values = np.append(cat_0_values, 1/np.power(indice_values[1], self.weight_power))
                    elif int(indice_values[0]) == 1:
                        cat_1_values = np.append(cat_1_values, 1/np.power(indice_values[1], self.weight_power))
                    else:
                        cat_2_values = np.append(cat_2_values, 1/np.power(indice_values[1], self.weight_power))
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
        if self.distance_measurements_method == "euclidean":
            self.updateDistancesEuclidean()
        else:
            self.updateDistancesManhattan()

        if self.distance_weighted:
            self.predictCategoriesWeighted()
        else:
            self.predictCategoriesBasic()

        return self.predictionAccuracy()

#Euclidean, Manhattan
values = np.array([])
for k in range(1,11):
    ml = MLPart2a("data\\classification\\trainingData.csv", "data\\classification\\testData.csv", k, True, "Euclidean", 2)
    prediction = ml.search()
    values = np.append(values, prediction)
    print("Prediction accuracy: ", prediction)


