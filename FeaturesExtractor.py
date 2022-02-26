from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim.downloader as api
import numpy as np

"""Factory method for the creation of the features extractor"""
class FeaturesExtractor(ABC):
    
    def __init__(self, train):
        self._train = train
        
    def setModelMode(self, train):
        self._train = train
                
    def getModelMode(self):
        return self._train
    
    @staticmethod
    def getAlgorithm(algorithm):
        if algorithm == "tfidf":
            return ConcreteFeaturesExtractorTFIDF().factoryMethod()
        elif algorithm == "doc2vec":
            return ConcreteFeaturesExtractorDoc2Vec().factoryMethod()

    @abstractmethod
    def factoryMethod(self):
        pass       
    

class ConcreteFeaturesExtractorTFIDF(FeaturesExtractor):
    
    def __init__(self, train = True):
        super().__init__(train)
    
    def factoryMethod(self):
        return ConcreteAlgorithmTFIDF(self._train)

class ConcreteFeaturesExtractorDoc2Vec(FeaturesExtractor):
    
    def __init__(self, train = True):
        super().__init__(train)
    
    def factoryMethod(self):
        return ConcreteAlgorithmDoc2Vec(self._train)
  

class Algorithm(ABC):
    
    def __init__(self, train = True):
        self._train = train
        self._dataX = None
        
    def setDataX(self, dataX):
        self._dataX = dataX
        
    def getDataX(self):
        return self._dataX
    
    def setModelMode(self, train):
        self._train = train
                
    def getModelMode(self):
        return self._train
    
    @abstractmethod
    def run(self, dataX):
        pass
    
class ConcreteAlgorithmTFIDF(Algorithm):
    
    def __init__(self, train = True):
        super().__init__(train)
        self.__counterVectorizer = CountVectorizer()
        self.__transformer = TfidfTransformer()
        self.__featuresTrain = self.__featuresTest = self.__vocabularyTrain = self.__vocabularyTest = None

    """From tokens to sentences"""        
    def __adaptData(self):
        data = list()
        nDocs = len(self._dataX)
        for i in range(nDocs):
            nWords = len(self._dataX[i])
            doc = ""
            for j in range(nWords):
                doc += (self._dataX[i][j] + " ")
            data.append(doc)
        self._dataX = data

    """Counting words frequencies""" 
    def __createVocabulary(self):
        if self._train:
            self.__vocabularyTrain = self.__counterVectorizer.fit_transform(self._dataX)
        else:
            self.__vocabularyTest = self.__counterVectorizer.transform(self._dataX)
    
    """Create bag of words model"""   
    def __createBoW(self):
        if self._train:
            self.__featuresTrain = self.__transformer.fit_transform(self.__vocabularyTrain)
        else:
            self.__featuresTest = self.__transformer.fit_transform(self.__vocabularyTest)
    
    def getFeatures(self):
        if self._train:
            return self.__featuresTrain.toarray()
        else:
            return self.__featuresTest.toarray()
        
    def getVocabulary(self):
        if self._train:
            return self.__vocabularyTrain.toarray()
        else:
            return self.__vocabularyTest.toarray()
    
    def run(self, dataX):
        self.setDataX(dataX)
        self.__adaptData()
        self.__createVocabulary()
        self.__createBoW()
        return 'TF.IDF done!'
   
class ConcreteAlgorithmDoc2Vec(Algorithm):
    
    def __init__(self, train):
        super().__init__(train)
        if self._train:
            self.__d2v = Doc2Vec(vector_size = 1024, dm = 0, dbow_words = 0, alpha = 0.025, min_count = 1, window = 7)
        else:
        	self.__d2v = Doc2Vec.load('doc2vec.bin')
 
    def __createVocabulary(self):
        self.__taggedData = [TaggedDocument(words = _d, tags = [str(i)]) for i, _d in enumerate(self._dataX)]
        self.__d2v.build_vocab(self.__taggedData)
    
    def __trainModel(self):
        for epoch in range(30):
            print('Iteration {}'.format(epoch))
            self.__d2v.train(self.__taggedData, total_examples = self.__d2v.corpus_count, epochs = self.__d2v.iter)
        
    def __saveModel(self):
        self.__d2v.save('d2v.model')
        
    def __getDocVector(self, doc):
        return self.__d2v.infer_vector(doc)
        
    def getFeatures(self):
        features = []
        for doc in self._dataX:
            features.append(self.__getDocVector(doc))
        return np.asarray(features)
            
    def run(self, dataX):
        self.setDataX(dataX)
        if self._train:
            self.__createVocabulary()
            self.__trainModel
            self.__saveModel()
        return 'Doc2Vec done'
