import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from collections import Counter


class Dataset:
    
    def __init__(self):
        self.__dataPath = "./dataset+additional_analysis.xlsx"
        self.__noClasses = ["Add issue", "Add issue ", "info release issue", "info release issue "]
        
    def __readXLS(self):
        self.__dataframe = pd.read_excel(self.__dataPath, sheet_name = 0)
    
    """drop empty extra columns and remove 2 classes not used in the classification process"""
    def __cleanData(self):
        self.__dataframe.drop(self.__dataframe.columns[6:], axis = 1, inplace = True)
        for noCl in self.__noClasses:
            indexNames = self.__dataframe[self.__dataframe['Classification'] == noCl].index
            self.__dataframe.drop(indexNames, inplace = True)
        self.__dataframe.dropna(axis = 0, inplace = True)
    
    """delete redundant class names due to extra spaces"""       
    def __parseClasses(self, classes):
        newClasses = list()
        for i in range(len(classes)):
            if classes[i][-1] != ' ':
                newClasses.append(classes[i])
        return newClasses

    """extract class names"""            
    def __extractClasses(self):
        classes = list(self.__dataframe['Classification'].unique())
        classes = self.__parseClasses(classes)
        self.__classes = dict(zip(range(0, len(classes)), classes))
    
    """extract labels"""        
    def __mapToClass(self, name):
        for key, value in self.__classes.items():
            if value == name[:len(value)]:
                return key
    
    def __extractData(self):
        self.__dataX = self.__dataframe['Summary'].to_numpy()
        self.__extractClasses()
        dataY = self.__dataframe['Classification'].tolist()
        for i in range(len(dataY)):
            dataY[i] = self.__mapToClass(dataY[i])
        self.__dataY = np.array(dataY)

    """plot class distributions"""        
    def __countingData(self, train = 0):
        data = self.__dataTrainY 
        if train  == 1:
            data = self.__dataValidationY
        elif train == 2:
            data = self.__dataTestY
        self.__classCounters = Counter(data)
        self.__classCounters = {k:v for k, v in sorted(self.__classCounters.items(), key = lambda item: item[0])}
        counters = list(self.__classCounters.values())
        labels = list(self.__classes.values())
        plt.pie(counters, labels = labels, autopct = '%1.1f%%', radius = 2)
        titlePlot = 'Training set'
        if train == 1:
            titlePlot = 'Validation set'    
        elif train == 2:
            titlePlot = 'Testing set'
        plt.title(titlePlot, bbox={'facecolor':'0.8', 'pad':5}, y = 1.45)
        plt.show()
    
    """splitting data in training-validation-testing set"""        
    def __splittingData(self):
        self.__dataTrainX, self.__dataTestX, self.__dataTrainY, self.__dataTestY = train_test_split(self.__dataX, self.__dataY, test_size = .25, random_state = 1702, shuffle = True)
        self.__dataTrainX, self.__dataValidationX, self.__dataTrainY, self.__dataValidationY = train_test_split(self.__dataTrainX, self.__dataTrainY, test_size = .10, random_state = 1702, shuffle = True)
    
    def setDataX(self, dataX):
        self.__dataX = dataX
        
    def setDataY(self, dataY):
        self.__dataY = dataY
        
    def setDataTrainX(self, dataTrainX):
        self.__dataTrainX = dataTrainX
        
    def setDataTestX(self, dataTestX):
        self.__dataTestX = dataTestX
        
    def setDataTrainY(self, dataTrainY):
        self.__dataTrainY = dataTrainY
        
    def setDataTestY(self, dataTestY):
        self.__dataTestY = dataTestY
        
    def setDataPath(self, dataPath):
        self.__dataPath = dataPath
        
    def setDataframe(self, dataframe):
        self.__dataframe = dataframe
    
    def setNoClasses(self, noClasses):
        self.__noClasses = noClasses
        
    def setClasses(self, classes):
        self.__classes = classes
    
    def getDataX(self):
        return self.__dataX
    
    def getDataY(self):
        return self.__dataY
    
    def getDataTrainX(self):
        return self.__dataTrainX
    
    def getDataTestX(self):
        return self.__dataTestX
    
    def getDataTrainY(self):
        return self.__dataTrainY
    
    def getDataTestY(self):
        return self.__dataTestY

    def getDataValidationX(self):
        return self.__dataValidationX

    def getDataValidationY(self):
        return self.__dataValidationY
    
    def getDataPath(self):
        return self.__dataPath
    
    def getDataframe(self):
        return self.__dataframe
    
    def getNoClasses(self):
        return self.__noClasses
    
    def getClasses(self):
        return self.__classes
    
    def getClassCounters(self):
        return self.__classCounters
    
    def run(self):
        self.__readXLS()
        self.__cleanData()
        self.__extractData()
        self.__splittingData()
        self.__countingData()
        self.__countingData(1)
        self.__countingData(2)
    
    
        
            
    
    
