import sys
import numpy as np
import random
from matplotlib import pyplot as plt

def euclidean_distance(center, point):
    sum = 0
    for i in range(len(center)):
        sum += (center[i]-point[i])**2
    return np.sqrt(sum)

class KMeans():
    def __init__(self, k, iterations):
        self.k = k
        self.iterations = iterations

    def fit(self, data):
        self.centers = self.initialize_random_centers(data)
        self.clusters, self.clustered_data = self.clustering(data)

        for i in range(self.iterations):
            self.centers = self.mean_centers()
            self.clusters, self.clustered_data = self.clustering(data)

        return self.clustered_data

    def initialize_random_centers(self, data):
        dim = len(data[0])
        centers = []
        for i in range(self.k):
            center = []
            for j in range(dim):
                X = np.array(data)[:, j]
                rand = random.uniform(min(X), max(X))
                center.append(rand)
            centers.append(center)
        return centers

    def clustering(self, data):
        clustered_data = {}
        clusters = {}
        for i in range(len(self.centers)):
            clusters[i] = []

        for point_idx in range(len(data)):
            point = data[point_idx]
            nearest_center_idx, nearest_center_dist = None, float('inf')
            for i in range(len(self.centers)):
                center = self.centers[i]
                dist = euclidean_distance(center, point)
                if nearest_center_idx == None or nearest_center_dist > dist:
                    nearest_center_dist = dist
                    nearest_center_idx = i

            # clusters: center_idx -> list of points
            clusters[nearest_center_idx].append(point)
            # clustered_data: point_idx -> center
            clustered_data[point_idx] = center # izabciti?

        return clusters, clustered_data

    def mean_centers(self):
        new_centers = []
        for i in range(len(self.centers)):
            current_center = self.centers[i]
            center = np.zeros((len(current_center),), dtype=float)
            cluster = self.clusters[i]
            for point in cluster:
                for i in range(len(point)):
                    center[i] += point[i]
            if len(cluster) > 0:
                new_centers.append(center/len(cluster))
            else:
                new_centers.append(current_center)
        return new_centers

    def predict(self, point):
        if len(point) != len(self.centers[0]):
            raise ValueError('Incorrect point dimensions.')

        nearest_center_idx, nearest_center_dist = None, float('inf')
        for i in range(len(self.centers)):
            center = self.centers[i]
            dist = euclidean_distance(center, point)
            if nearest_center_idx == None or nearest_center_dist > dist:
                nearest_center_dist = dist
                nearest_center_idx = i
            
        return nearest_center_idx

def cluster_visualization(file_path, clusters, centers):
    plt.figure()
    for i in range(len(clusters)):
        data = np.array(clusters[i])
        center = centers[i]
        if len(data) > 0:
            plt.scatter(data[:, 0], data[:, 1])
        plt.scatter(center[0], center[1], marker='X', color='black')
    plt.savefig(file_path + 'clustered_plot.png')

def main():
    return

if __name__ == '__main__':
    main()