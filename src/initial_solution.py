import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import pandas as pd
import random

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
    plt.close()

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

def randomized_initialization(data, k, output_dir_path, algorithm_name):
    columns = ['x', 'y']
    instances = data[columns]
    scaler = MinMaxScaler()
    instances = pd.DataFrame(scaler.fit_transform(instances), columns=columns)

    inits = ['k-means++', 'random']
    n_inits = [10, 20, 'auto']
    tols = [1e-5, 1e-4, 1e-3]
    random_states = [0, 42, None]
    algoritms = ['lloyd', 'elkan']

    init = random.choice(inits)
    n_init = random.choice(n_inits)
    tol = random.choice(tols)
    random_state = random.choice(random_states)
    algorithm = random.choice(algoritms)

    model = KMeans(n_clusters=k, init=init, n_init=n_init, tol=tol, random_state=random_state, algorithm=algorithm)
    model.fit(instances)

    regr_coefs, regr_intercept = elastic_net(data, model.labels_, k)
    error, mse = regression_error(data, model.labels_, regr_coefs, regr_intercept, k)
    nearest_clusters = calculate_nearest_clusters(data, regr_coefs, regr_intercept, k)
    params = f'{init}_{n_init}_{tol}_{random_state}_{algorithm}'

    return model.labels_, regr_coefs, regr_intercept, error, mse, nearest_clusters, params

def main():
    return

if __name__ == '__main__':
    main()