from imblearn.over_sampling import SMOTE, ADASYN, KMeansSMOTE
import numpy as np
from Classifier import ClassifierCreator

"""Dataset balancing"""
class Balance:
    def __init__(self, seed = 1702, neighbors = 5):
        """to individuate new sub classes
        self.__classifier = ClassifierCreator().getClassifier('kmeans') 
        """
    
        self.__seed = seed
        self.__neighbors = neighbors
        #self.__sm = SMOTE(random_state = self.__seed, k_neighbors = self.__neighbors)
        self.__sm = ADASYN(random_state = self.__seed, n_neighbors = self.__neighbors)
        
    def setSeed(self, seed):
        self.__seed = seed
        #self.__sm = SMOTE(random_state = self.__seed, k_neighbors = self.__neighbors)

    def setNeighbors(self, neighbors):
        self.__neighbors = neighbors
        #self.__sm = SMOTE(random_state = self.__seed, k_neighbors = self.__neighbors)


    def run(self, dataX, dataY, mode='train'):
        """Decomment the algorithm that you want to use"""
        #return self.__sm.fit_resample(dataX, dataY)
        """ Balance with crossing over operations.
        return self.population_sp_crossover(dataX, dataY)
        """
        
        """to individuate new sub classes
        new_labels = [9,2,10,11]
        elements_of_class_2 = np.where(dataY == 2)[0]
        
        if mode == 'train':
            self.__classifier.classifier = self.__classifier.classifier.fit(dataX[elements_of_class_2])
            
        predictions = self.__classifier.classifier.predict(dataX[elements_of_class_2])

        for i in range(4):
            el_per_classes = np.where(predictions == i)[0]

            if len(el_per_classes) == 0:
                continue
            dataY[elements_of_class_2[el_per_classes]] = new_labels[i]
        """
        if mode == 'train':
            dataX, dataY = self.__sm.fit_resample(dataX, dataY)
        
        return dataX, dataY

    """implement single point crossover"""
    def sp_crossover(self, p1, p2, point=None):
        #Implement bi point crossover between two parents: p1, p2
        first_son = p1.copy()
        second_son = p2.copy()

        if point is None:
            first_son[first_son.shape[0]//2:] = p2[first_son.shape[0]//2:]       
            second_son[first_son.shape[0]//2:] = p1[first_son.shape[0]//2:]
        else:
            first_son[point:] = p2[point:]       
            second_son[point:] = p1[point:]
        
        return first_son, second_son
    
    """implement single point crossover for a set of documents"""
    def population_sp_crossover(self, features, labels):
        features = features
        
        for classes in range(9):
            if classes == 2:
                continue
            parents = np.where(labels == classes)[0]
            
            #np.random.shuffle(parents)

            for i in range(0,parents.shape[0]-1,2):
                son, second_son = self.sp_crossover(features[parents[i]], features[parents[i+1]])

                son = son.reshape(1,-1)
                second_son = second_son.reshape(1,-1)

                features = np.concatenate((features, son), axis=0)
                features = np.concatenate((features, second_son), axis=0)
                labels = np.append(labels, classes)
                labels = np.append(labels, classes)
            
        return features, labels
