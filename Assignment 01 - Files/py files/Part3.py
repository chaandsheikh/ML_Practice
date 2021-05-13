import numpy as np

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
        self.originalValues = np.array([])
        self.readInstances()
        self.k = __k
        #self.accuracy

    def readInstances(self):
        """
        Reading an instance from fName
        """
        self.trainingData = np.genfromtxt(self.trainingDataFileName, delimiter=',', dtype=float)
        self.testData = np.genfromtxt(self.testDataFileName, delimiter=',', dtype=float)
        self.originalValues = self.testData[:, -1]

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
         self.euclideanDistances = np.array(distances)
         self.euclideanDistancesSorted = np.array(sortedDistances)

    def predictValue(self):
        indices = self.euclideanDistancesSorted[:, :self.k]
        values_of_neighbours = self.trainingData[indices, -1]
        #print(np.shape(values_of_neighbours))
        neighbours_euclideanDistances = self.euclideanDistances[:, indices[0]]
        neighbours_distances_inverse_squared = 1/np.square(neighbours_euclideanDistances)
        for i in range(0,self.testData.shape[0]):
            self.predictions = np.append(self.predictions, np.sum(neighbours_distances_inverse_squared[i,:] * values_of_neighbours[i,:])/np.sum(neighbours_distances_inverse_squared[i,:]))

    def predictionAccuracy(self):
        prediction = np.array([])
        originalValues = self.testData[:, -1]
        numerator = np.sum(np.square(self.predictions - self.testData[:, -1]))
        denominator = np.sum(np.square(np.average(self.testData[:, -1]) - self.testData[:, -1]))
        accuracy = 1 - (numerator/denominator)
        return accuracy * 100


    def search(self):
        self.updateDistances()
        self.predictValue()
        return self.predictionAccuracy()


ml = MLPart1("RtrainingData.csv", "RtestData.csv", 8)
print("Prediction accuracy : ", ml.search())


