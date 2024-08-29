import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import pandas as pd

from utils import cluster_visualization
from utility_functions import elastic_net
from algorithm import regression_error, calculate_nearest_clusters

def simple_visualization(data, file_path):
    X = data['x']
    y = data['y']
    plt.scatter(X, y)
    plt.xlabel('x')
    plt.xlabel('y')
    plt.title('Initial data plot')
    plt.savefig(file_path)

def initialization(data, k, output_dir_path, algorithm_name):
    dir_path = os.path.join(output_dir_path, algorithm_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = os.path.join(dir_path, 'initial_plot.png')    
    simple_visualization(data, file_path)

    columns = ['x', 'y']
    instances = data[columns]
    scaler = MinMaxScaler()
    instances = pd.DataFrame(scaler.fit_transform(instances), columns=columns)

    model = KMeans(n_clusters=k, n_init='auto')
    model.fit(instances)

    regr_coefs, regr_intercept = elastic_net(data, model.labels_, k)
    error, mse = regression_error(data, model.labels_, regr_coefs, regr_intercept, k)
    nearest_clusters = calculate_nearest_clusters(data, regr_coefs, regr_intercept, k)

    centers = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=columns)
    cluster_visualization(
        data, 
        centers, 
        model.labels_, 
        k, 
        regr_coefs,
        regr_intercept,
        output_dir_path,
        algorithm_name, 
        'initial_clustered_plot.png'
    )

    return model.labels_, regr_coefs, regr_intercept, error, mse, nearest_clusters

def main():
    return

if __name__ == '__main__':
    main()