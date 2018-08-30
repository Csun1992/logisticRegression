import numpy as np
from math import exp, log, sqrt
from copy import copy
import sys

def sign(x):
    if np.sign(x)==0:
        return -1
    else:
        return int(np.sign(x))

def findLine(point1, point2):
    slope = (point2[1]-point1[1])/float(point2[0]-point1[0])
    intercept = point2[1] - slope*point2[0]
    return slope, intercept

def classify(data, slope, intercept):
    group = []
    size = np.size(data, 0)
    for i in range(size):
        group.append(sign(data[i,1]-slope*data[i,0]-intercept))
    return group

def findTestErr(testData, weight):
    size = np.size(testData, 0)
    error = 0
    for i in range(size):
        x = testData[i, :-1].reshape(-1, 1)
        y = testData[i, -1]
        error += log(1+exp(-y*weight.T.dot(x))) 
    return error/size



class StochasticGradient(object):
    def __init__(self, data):
        self.data = data
        self.sampleSize = np.size(data, 0)
        self.dataDim = np.size(data, 1) - 1 
        self.weight = np.zeros((self.dataDim, 1))
        self.epoch = 0
        self.learningRate = 0.01
        self.tolerance = 0.01

    def setLearningRate(self, rate):
        self.learningRate = rate
        return self.learningRate

    def setTolerance(self, tolerance):
        self.tolerance = tolerance
        return self.tolerance

    def gradient(self, index):
        x = self.data[index, :-1].reshape(-1, 1)
        y = self.data[index, -1]
        grad = -y*x/(1+exp(y*self.weight.T.dot(x)))
        return grad

    def updateWeight(self):
        index = np.random.permutation(range(self.sampleSize))
        for i in index:
            self.weight -= self.learningRate*self.gradient(i)
        return self.weight

    def descent(self):
        diff = float("inf") 
        while diff > self.tolerance:
            weight = copy(self.weight)
            self.updateWeight()
            diff = sqrt((self.weight-weight).T.dot(self.weight-weight))
            self.epoch += 1
        return self.epoch

    def getWeight(self):
        return self.weight

    def getEpoch(self):
        return self.epoch




if __name__ == '__main__':
    experimentNum = 1
    dataDim = 2
    lowerLim = -1
    upperLim = 1

    # experiment with sample size = 100
    sampleSize = 100
    totalEpoch = 0
    error = 0

    for i in range(experimentNum):
        x1 = np.random.uniform(lowerLim, upperLim, 2).reshape(-1, 1)
        x2 = np.random.uniform(lowerLim, upperLim, 2).reshape(-1, 1)
        slope, intercept = findLine(x1, x2)
        data = np.random.uniform(lowerLim, upperLim, sampleSize*dataDim).reshape(sampleSize, dataDim)
        group = np.array(classify(data, slope, intercept)).reshape(-1, 1)
        biasTerm = np.ones(sampleSize).reshape(-1, 1)
        data = np.concatenate((biasTerm, data, group), axis=1)
        stochasticGradient = StochasticGradient(data)
        stochasticGradient.descent()
        totalEpoch += stochasticGradient.getEpoch()

        testData = np.random.uniform(lowerLim, upperLim, sampleSize*dataDim).reshape(sampleSize, dataDim)
        group = np.array(classify(testData, slope, intercept)).reshape(-1, 1)
        testData = np.concatenate((biasTerm, testData, group), axis=1)
        error += findTestErr(testData, stochasticGradient.getWeight())


    print "out sample error is given by:"
    print error/experimentNum

    print "total epoch is:"
    print totalEpoch/float(experimentNum)

