from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import ElasticNet
import pandas as pd

from utils import cluster_visualization

def initial_solution(data, k, output_file_path):
    columns = ['x', 'y']
    instances = data[columns]
    scaler = MinMaxScaler()
    instances = pd.DataFrame(scaler.fit_transform(instances), columns=columns)

    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(instances)
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=columns)
    cluster_visualization(instances, centers, kmeans.labels_, k, output_file_path, 'initial_clustered_plot.png')

    regr_coefs = []
    regr_intercept = []
    for c in range(centers.shape[0]):
        cluster = data[kmeans.labels_ == c]
        regr = ElasticNet(random_state=0)
        y = cluster.loc[:, 'y'].values
        X = cluster.loc[:, 'x'].values.reshape((y.shape[0], 1))
        regr.fit(X, y)
        regr_coefs.append(regr.coef_)
        regr_intercept.append(regr.intercept_)

    return kmeans.labels_, regr_coefs, regr_intercept

def main():
    return

if __name__ == '__main__':
    main()