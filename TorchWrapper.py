from torch.utils.data.dataset import Dataset as DatasetTorch
import numpy as np
from copy import copy


"""to convert the dataset to the pytorch useful format"""
class TorchWrapper(DatasetTorch):
    def __init__(self):
        self.__features_size = 2048

    def setData(self, dataX, dataY):
        self.__dataX = dataX
        self.__dataX = self.__addPad()
        self.__dataY = dataY

    def getData(self):
        return copy(self), copy(self)

    """Padding the features representation. To use with the tfidf features extractor"""
    def __addPad(self):
        pad_size = (self.__dataX.shape[0], self.__features_size)
        padded_data = np.zeros(pad_size)
        padded_data[:self.__dataX.shape[0],:self.__dataX.shape[1]] = self.__dataX
        
        return padded_data

    def __getitem__(self, index):
        return self.__dataX[index], self.__dataY[index]

    def __len__(self):
        return len(self.__dataX)