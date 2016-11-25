from abc import ABCMeta, abstractmethod

class Label:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __str__(self): pass
       
class ClassificationLabel(Label):
    def __init__(self, label):
        self.label = label
        
    def __str__(self):
        return str(self.label)

class FeatureVector:
    def __init__(self):
        self.feature_vector = {}
        
    def add(self, index, value):
        self.feature_vector[index] = value
                
    def get(self, index):
        if self.feature_vector.get(index) != None:
            return self.feature_vector.get(index)
        else:
            return 0.0
        
class Instance:
    def __init__(self, feature_vector, label):
        self._feature_vector = feature_vector
        self._label = label

class Predictor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, instances): pass

    @abstractmethod
    def predict(self, instance): pass

       
