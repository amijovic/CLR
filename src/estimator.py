import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator

from utility_functions import calc_cluster_centroids, dist_to_regression_line
from vnd import vnd
from grasp import grasp

class Estimator(BaseEstimator):
    def __init__(self, k, algorithm, attrib_columns, target):
        super().__init__()
        self.k = k
        self.algorithm = algorithm
        self.attrib_columns = attrib_columns
        self.target = target
        
    def fit(self, X, y):
        data = pd.DataFrame(X, columns=self.attrib_columns)
        data[self.target] = y
        self.data = data

        if self.algorithm == 'grasp':
            self.mse, self.regr_coefs, self.regr_interception, self.labels, self.params = grasp(
                data,
                self.k
            )
        else:
            self.mse, self.regr_coefs, self.regr_interception, self.labels = vnd(
                data,
                self.k,
                self.algorithm
            )

    def predict(self, X):
        # # centroids = calc_cluster_centroids(X, self.k, self.labels_)
        # predictions = [self._distance_prediction(x) for x in X]
        # return predictions
        w = self._calc_weighs()
        predictions = [self._weighed_prediction(x, w) for x in X]
        return predictions
    
    def _distance_prediction(self, x):
        y_pred = -1
        dist_to_nearest_cluster = float('inf')
        
        for c in range(self.k):
            y = np.dot(x, self.regr_coefs[c]) + self.regr_interception[c]
            point = np.append(x, y)
            dist_to_centroid = dist_to_regression_line(point, self.regr_coefs[c], self.regr_interception[c]) 
            if dist_to_nearest_cluster > dist_to_centroid:
                y_pred = y
                dist_to_nearest_cluster = dist_to_centroid
        return y_pred
    
    def _calc_weighs(self):
        w = []
        for j in range(self.k):
            cluster_size = len(self.data[self.labels == j])
            wj = cluster_size / self.data.shape[0]
            w.append(wj)
        return w

    def _weighed_prediction(self, x, w):
        y_pred = 0
        for c in range(self.k):
            y = np.dot(x, self.regr_coefs[c]) + self.regr_interception[c]
            y_pred += w[c]*y
        return y_pred