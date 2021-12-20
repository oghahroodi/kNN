from utils import *
import math

def normalize_data(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

class kNN:
    def __init__(self, k):
        self.k = k

    def distance(self, x1, x2):
        return math.sqrt(np.sum((x1-x2)**2))

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        
    def predict(self, x):
        dist_list = []
        for i in range(len(self.X)):
            # if (self.distance(x, self.X[i])==0):
            #     return self.Y[i]
            dist_list.append(self.distance(x, self.X[i]))
        k_neighbor = [i[0] for i in sorted(enumerate(dist_list), key=lambda a:a[1])][:self.k]
        s = 0
        for i in k_neighbor:
            s+=self.Y[i]
        if (s>=self.k/2):
            return 1
        else:
            return 0
