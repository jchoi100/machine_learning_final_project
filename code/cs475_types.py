from abc import ABCMeta, abstractmethod

class Instance:
    def __init__(self, feature_vector, label):
        self._feature_vector = feature_vector
        self._label = label

class Predictor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, feature_vectors, labels): pass

    @abstractmethod
    def predict(self, feature_vectors, labels): pass

       
