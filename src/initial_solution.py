import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import ElasticNet
import pandas as pd

from utils import read_data_from_file, simple_visualization ,cluster_visualization

def initial_solution(data, k):
    columns = ['x', 'y']
    instances = data[columns]
    scaler = MinMaxScaler()
    instances = pd.DataFrame(scaler.fit_transform(instances), columns=columns)

    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(instances)
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=columns)
    cluster_visualization(output_file_path, instances, centers, kmeans.labels_)

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

    return kmeans.cluster_centers_, kmeans.labels_, regr_coefs, regr_intercept

def main(input_file_path, output_file_path, k):
    data = pd.DataFrame(read_data_from_file(input_file_path), columns=['x', 'y'])
    simple_visualization(output_file_path, '/initial_plot.png', data)
    cluster_centers, labels, regr_coefs, regr_intercept = initial_solution(data, k)

    print(f'centers: \n{cluster_centers}')
    print(f'labels: \n{labels}')
    print(f'coefs: \n{regr_coefs}')
    print(f'intercept: \n{regr_intercept}')

# python3 test.py data_file_path output_file_path k
if __name__ == '__main__':
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    k = int(sys.argv[3])
    main(input_file_path, output_file_path, k)