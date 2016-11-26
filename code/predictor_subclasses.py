from cs475_types import Predictor
from math import sqrt
from scipy.spatial.distance import euclidean

""" HW3 """
class KNN(Predictor):

    def __init__(self, knn):
        self.K = knn
        self.train_set = []
        self.train_labels = []

    def train(self, train_set, train_labels):
        self.train_set = train_set
        self.train_labels = train_labels

    def predict(self, test_vector):
        neighbors = []
        for i in range(len(self.train_set)):
            x_i = self.train_set[i]
            y_i = self.train_labels[i]
            distance = euclidean(x_i, test_vector)
            neighbors.append((y_i, distance))
        neighbors = sorted(neighbors, key=lambda tup: (tup[1], tup[0]))
        nearest_neighbors = neighbors[0:self.K]
        votes = {}
        for neighbor in nearest_neighbors:
            if votes.has_key(neighbor[0]):
                votes[neighbor[0]] -= 1
            else:
                votes[neighbor[0]] = -1
        votes = sorted(votes.items(), key=lambda tup: (tup[1], tup[0]))
        print votes
        return votes[0][0]
