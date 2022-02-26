from Dataset import *
from Preprocessing import *
from FeaturesExtractor import *
from Classifier import *
from TorchWrapper import *
from Balance import *
import argparse
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

"""Software provides two mode of using it: TRAIN mode to the learning phase of the classification; TEST mode for
the evalutation"""

TEST = 0
TRAIN = 1


class BugsClassification:
    
    def __init__(self, featureExtractor, classifier):
        self.__dataset = Dataset()
        self.__preprocessing = PreProcessing()
        self.__featuresExtractor = FeaturesExtractor.getAlgorithm(featureExtractor)
        self.__balance = Balance()
        self.__torchWrapper = TorchWrapper()
        if classifier == 'mlp':
            self.__torch = True
        else:
            self.__torch = False
        if featureExtractor == 'doc2vec' or featureExtractor == 'word2vec':
            self.__model = True
        else:
            self.__model = False
        self.__classifier = ClassifierCreator().getClassifier(classifier)
        self.__mode = TRAIN
        
    """Training phase"""
    def __training(self):
        #from features extractor class
        print('Extracting features...')
        self.__featuresExtractor.setModelMode(True)
        self.__featuresExtractor.run(self.__tokenTrainX)
        if self.__model:
            self.__featuresExtractor.setModelMode(False)
            self.__featuresExtractor.run(self.__tokenTrainX)
        self.__featuresTrainX = self.__featuresExtractor.getFeatures()
        
        self.__featuresExtractor.setModelMode(False)
        self.__featuresExtractor.run(self.__tokenValidationX)
        self.__featuresValidationX = self.__featuresExtractor.getFeatures()
        #print(self.__featuresValidationX.shape)
        
        #from balance class
        print('Balancing data...')
        self.__featuresTrainX, self.__dataTrainY = self.__balance.run(self.__featuresTrainX, self.__dataTrainY)
        #self.__featuresValidationX, self.__dataValidationY = self.__balance.run(self.__featuresValidationX, self.__dataValidationY, mode='test')

        #from torch wrapper class
        print('Training classifier...')
        if self.__torch:
            self.__torchWrapper.setData(self.__featuresTrainX, self.__dataTrainY)
            self.__featuresTrainX, self.__dataTrainY = self.__torchWrapper.getData()
            
            self.__torchWrapper.setData(self.__featuresValidationX, self.__dataValidationY)
            self.__featuresValidationX, self.__dataValidationY = self.__torchWrapper.getData()
        
        #from classifier class
        self.__classifier.train(self.__featuresTrainX, self.__dataTrainY, self.__featuresValidationX, self.__dataValidationY)

    """Testing phase"""        
    def __evaluate(self):
        #from features extractor class
        print('Extracting features...')
        self.__featuresExtractor.setModelMode(False)
        self.__featuresExtractor.run(self.__tokenTestX)
        self.__featuresTestX = self.__featuresExtractor.getFeatures()
        
        self.__featuresTestX, self.__dataTestY = self.__balance.run(self.__featuresTestX, self.__dataTestY, mode='test')
        
        #from torch wrapper class
        print('Testing classifier...')
        if self.__torch:
            self.__torchWrapper.setData(self.__featuresTestX, self.__dataTestY)
            self.__featuresTestX, self.__dataTestY = self.__torchWrapper.getData()
        
        #from classifier class
        self.__score = self.__classifier.evaluate(self.__featuresTestX, self.__dataTestY)
        
    def setMode(self, mode):
        self.__mode = mode

    def getMode(self):
        return self.__mode
    
    def getScore(self):
        return self.__score
    
    """Getting data from dataset"""
    def getData(self):
        #from dataset class
        print('Reading data...')
        self.__dataset.run()
        self.__dataTrainX = self.__dataset.getDataTrainX()
        self.__dataTrainY = self.__dataset.getDataTrainY()
        self.__dataValidationX = self.__dataset.getDataValidationX()
        self.__dataValidationY = self.__dataset.getDataValidationY()
        self.__dataTestX = self.__dataset.getDataTestX()
        self.__dataTestY = self.__dataset.getDataTestY()
        
        #from preprocessing class
        print('Preproccesing data...')
        self.__preprocessing.run(self.__dataTrainX)
        self.__tokenTrainX = self.__preprocessing.getDoc()
        self.__preprocessing.run(self.__dataValidationX)
        self.__tokenValidationX = self.__preprocessing.getDoc()
        self.__preprocessing.run(self.__dataTestX)
        self.__tokenTestX = self.__preprocessing.getDoc()
        
    def getData2(self):
        print('Reading data...')
        self.__dataset.run()
        self.__dataTrainX = self.__dataset.getDataTrainX()
        self.__dataTrainY = self.__dataset.getDataTrainY()
        self.__dataValidationX = self.__dataset.getDataValidationX()
        self.__dataValidationY = self.__dataset.getDataValidationY()
        self.__dataTestX = self.__dataset.getDataTestX()
        self.__dataTestY = self.__dataset.getDataTestY()
        counter = Counter()
        tokenizer = get_tokenizer('basic_english')
        for line in self.__dataTrainX:
            counter.update(tokenizer(line))
        vocab = Vocab(counter, min_freq = 1)
        text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
        
    def run(self):
        if self.__mode == TRAIN:
            self.__training()
            return None
        self.__evaluate()
        return self.__score
            
            

        
def bugsClassification():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_extractor', type=str, help='Type of feature extractor: choose between tfidf or doc2vec')  
    parser.add_argument('--classifier', type=str, help='Type of classifier, choose one of these: knn, kmeans, logisticRegression, mlp') 
    
    opt = parser.parse_args()

    main = BugsClassification(opt.features_extractor, opt.classifier)
    
    main.getData()
    main.run()
    main.setMode(TEST)
    score = main.run()
    print(score)

if __name__ == '__main__':
    bugsClassification()