import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.stats import mode
from sklearn.linear_model import LogisticRegression
from Mlp import Mlp
import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
import os
from matplotlib import pyplot as plt

"""Classifier Creator with factory method"""
class ClassifierCreator():
    
    def __init__(self):
        super(ClassifierCreator, self).__init__()
    
    @staticmethod
    def getClassifier(classifier):
        if classifier == "knn":
            return KNNClassifier()
        elif classifier == "kmeans":
            return KMeansClassifier()
        elif classifier == "logisticRegression":
            return LogisticRegressionClassifier()
        elif classifier == "mlp":
            return MlpClassifier()          

"""Classifier class"""
class Classifier():
    def __init__(self):
        super(Classifier, self).__init__()
        
    def getMetrics(self, labels, predictions):
        accuracy_result = accuracy_score(labels, predictions)
        confusion_matrix_result = confusion_matrix(labels, predictions)
        classification_report_result = classification_report(labels, predictions)
        
        return accuracy_result, confusion_matrix_result, classification_report_result

    def train(self, training_set,  training_labels, validation_set=None, validation_labels=None):
        raise NotImplementedError("Please Implement this method")

    def evaluate(self, testing_set, testing_labels):
        raise NotImplementedError("Please Implement this method")

class KNNClassifier(Classifier):
    def __init__(self):
        super(KNNClassifier, self).__init__()
        self.classifier = KNeighborsClassifier(n_neighbors=7, algorithm='brute')

    def train(self, training_set,  training_labels, validation_set=None, validation_labels=None):
        self.classifier = self.classifier.fit(training_set, training_labels)

    def evaluate(self, testing_set, testing_labels):
        predictions = self.classifier.predict(testing_set)
        return self.getMetrics(testing_labels, predictions)

class LogisticRegressionClassifier(Classifier):
    def __init__(self):
        super(LogisticRegressionClassifier, self).__init__()
        self.classifier = LogisticRegression()

    def train(self, training_set,  training_labels, validation_set=None, validation_labels=None):
        self.classifier = self.classifier.fit(training_set, training_labels)

    def evaluate(self, testing_set, testing_labels):
        predictions = self.classifier.predict(testing_set)
        return self.getMetrics(testing_labels, predictions)

class KMeansClassifier(Classifier):
    def __init__(self):
        super(KMeansClassifier, self).__init__()
        self.n_classes = 9
        self.classifier = KMeans(init='random', n_clusters=self.n_classes, random_state=None, max_iter=1000)

    def train(self, training_set,  training_labels, validation_set=None, validation_labels=None):
        self.classifier = self.classifier.fit(training_set)

    def evaluate(self, testing_set, testing_labels):
        predictions = self.classifier.predict(testing_set)
        predictions = self.__mapping_labels(predictions, testing_labels)
        return self.getMetrics(testing_labels, predictions)
    
    def __mapping_labels(self, predictions, labels):
        new_predictions = predictions.copy()
        for i in range(self.n_classes):
            el_per_classes = np.where(predictions == i)[0]
            if len(el_per_classes) == 0:
                continue
            new_predictions[el_per_classes] = np.full(len(el_per_classes), self.__compute_mode(labels[el_per_classes]))    

        return new_predictions
    
    def __compute_mode(self, array):
        return np.array(mode(array)[0][0])

    def set_n_classes(self, n_classes):
        self.n_classes = n_classes
        self.classifier = KMeans(init='random', n_clusters=self.n_classes, random_state=15651, max_iter=1000)

class MlpClassifier(Classifier):
    
    def __init__(self):
        super(MlpClassifier, self).__init__()
        self.__classifier = Mlp()
        self.__criterion = nn.CrossEntropyLoss()
        #self.__optimizer = SGD(self.__classifier.parameters(), lr=0.001, momentum=0.99, weight_decay=0.0)
        self.__optimizer = Adam(self.__classifier.parameters(), lr=0.0001, weight_decay = 0.01)
        self.__best = 0
        self.__epochs = 100
        self.__exp_name = 'mlp'
        self.__logdir = 'logs'
    
    def __isBest(self, value):
        return True if value > self.__best else False
    
    """Plot training and validation curves"""
    def __plotLearning(self, train, validation):
        plt.plot(train, label = 'training')
        plt.plot(validation, label = 'validation')
        plt.legend(loc = 'lower right')
        plt.grid()
        #plt.yticks(np.arange(0, 1, step=0.1))
        plt.show()

    def train(self, training_set,  training_labels, validation_set=None, validation_labels=None):
        #Try to create a dir where we save weights
        try:
            os.makedirs(os.path.join(self.__logdir, self.__exp_name))
        except:
            pass

        #device = "cuda" if torch.cuda.is_available() else "cpu"

        #self.__classifier.to(device)

        training_set = DataLoader(training_set, batch_size=128, num_workers=8, shuffle=True)
        validation_set = DataLoader(validation_set, batch_size=128, num_workers=8, shuffle = True)

        loader = {
        'train' : training_set,
        'test' : validation_set,
        }
        
        acc_train = []
        acc_validation = []
        loss_train = []
        loss_validation = []

        for e in range(self.__epochs):
            for mode in ['train', 'test']:
                self.__classifier.train() if mode == 'train' else self.__classifier.eval()
                acc_mean = 0
                loss_mean = 0

                with torch.set_grad_enabled(mode == 'train'):
                    for i, batch in enumerate(loader[mode]):
                        x = batch[0]#.to(device) 
                        y = batch[1]#.to(device)
                        
                        output = self.__classifier(x.float())

                        l = self.__criterion(output, y)

                        if mode == 'train':
                            l.backward()
                            self.__optimizer.step()
                            self.__optimizer.zero_grad()
                        
                        acc = (accuracy_score(y.to('cpu'),output.to('cpu').max(1)[1]))
                        acc_mean = (acc_mean + acc)
                        loss_mean = (loss_mean + l.item())

                acc_mean = acc_mean/len(loader[mode])
                loss_mean = loss_mean/len(loader[mode])
                print("EPOCH: ", e, " MODE = ", mode, " ACC_MEAN = ", acc_mean, " LOSS_MEAN = ", loss_mean)
                if mode == 'train':
                    acc_train.append(acc_mean)
                    loss_train.append(loss_mean)
                else:
                    acc_validation.append(acc_mean)
                    loss_validation.append(loss_mean)
        
            torch.save({
                'weights': self.__classifier.state_dict()}, os.path.join(self.__logdir,self.__exp_name) + '/%s.tar'%(self.__exp_name))
            
            if self.__isBest(acc_mean):
                torch.save({
                'weights': self.__classifier.state_dict()}, os.path.join(self.__logdir,self.__exp_name) + '/%s_best.tar'%(self.__exp_name))
                
        self.__plotLearning(acc_train, acc_validation)
        self.__plotLearning(loss_train, loss_validation)

    def evaluate(self, testing_set, testing_labels):
        #device ="cuda" if torch.cuda.is_available() else "cpu"
        
        testing_set = DataLoader(testing_set, batch_size=128, num_workers=8, shuffle=True)
        
        #self.__classifier.to(device)
        
        self.__classifier.eval()
        
        predictions, labels = [], []
        
        with torch.set_grad_enabled(False):
            for i, batch in enumerate(testing_set):

                print("Processing batch:{}/{}".format( i, len(testing_set)))
                x = batch[0]#.to(device)
                y = batch[1]#.to(device)

                output = self.__classifier(x.float())

                preds = output.to('cpu').max(1)[1].numpy()

                predictions.extend(list(preds))
                labels.extend(list(y))

        predictions = np.array(predictions)
        labels = np.array(labels)
    
        return self.getMetrics(labels, predictions)

