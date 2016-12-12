from cs475_types import Predictor
from math import sqrt
from scipy.spatial.distance import euclidean
import numpy as np
from sklearn.svm import SVC
from td import tangentDistance

def td_kernel(X, Y):
    """
    """
    r1, c1 = X.shape
    r2, c2 = Y.shape
    zeros = np.zeros(c1)
    choice = np.ones(7)
    matrix = np.zeros((r1, r2))
    for r in range(r1):
        for c in range(r2):
            matrix[r][c] = 0.5 * (tangentDistance(X[r], zeros, 16, 16, choice)\
                                  + tangentDistance(Y[c], zeros, 16, 16, choice)\
                                  - tangentDistance(X[r], Y[c], 16, 16, choice))
    return matrix

class KNN(Predictor):

    def __init__(self, knn, is_svm, is_weighted=False):
        self.K = knn
        self.train_set = []
        self.train_labels = []
        self.is_svm = is_svm
        self.is_weighted = is_weighted

    def train(self, train_set, train_labels):
        self.train_set = train_set
        self.train_labels = train_labels

    def predict(self, test_vector):
        neighbors = []
        for i in range(len(self.train_set)):
            x_i = self.train_set[i]
            y_i = self.train_labels[i]
            distance = euclidean(x_i / 255.0, test_vector / 255.0)
            neighbors.append((y_i, distance, x_i))
        neighbors = sorted(neighbors, key=lambda tup: (tup[1], tup[0]))
        nearest_neighbors = neighbors[0:self.K]

        votes = {}
        if self.is_weighted:
            for neighbor in nearest_neighbors:
                if votes.has_key(neighbor[0]):
                    votes[neighbor[0]] -= 1.0 / (1 + neighbor[1]**2)
                else:
                    votes[neighbor[0]] = -1.0 / (1 + neighbor[1]**2)
            votes = sorted(votes.items(), key=lambda tup: (tup[1], tup[0]))
        else:
            for neighbor in nearest_neighbors:
                if votes.has_key(neighbor[0]):
                    votes[neighbor[0]] -= 1
                else:
                    votes[neighbor[0]] = -1
            votes = sorted(votes.items(), key=lambda tup: (tup[1], tup[0]))
            if self.is_svm:
                if -votes[0][1] == self.K:
                    return votes[0][0]
                else:
                    clf = SVC(kernel=td_kernel)
                    X = [x[2] for x in nearest_neighbors]
                    y = [x[0] for x in nearest_neighbors]
                    clf.fit(X, y)
                    print("!")
                    return clf.predict([test_vector])[0]
        return votes[0][0]
