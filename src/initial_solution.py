import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import pandas as pd
import random

from utility_functions import elastic_net
from algorithm import regression_error, calculate_nearest_clusters

def simple_visualization(data, file_path):
    X = data['x0']
    y = data['y']
    plt.scatter(X, y)
    plt.xlabel('x')
    plt.xlabel('y')
    plt.title('Initial data plot')
    plt.savefig(file_path)
    plt.close()

def initialization(data, k):
    columns = data.columns
    instances = data[columns]
    scaler = StandardScaler()
    instances = pd.DataFrame(scaler.fit_transform(instances), columns=columns)

    model = KMeans(n_clusters=k, n_init='auto')
    model.fit(instances)

    regr_coefs, regr_intercept = elastic_net(data, model.labels_, k)
    error, mse = regression_error(data, model.labels_, regr_coefs, regr_intercept, k)
    nearest_clusters = calculate_nearest_clusters(data, regr_coefs, regr_intercept, k)

    return model.labels_, regr_coefs, regr_intercept, error, mse, nearest_clusters

def initialization_find_k(data, output_dir_path, algorithm_name):
    solutions = []
    for k in range(2, 10):
        solution = initialization(data, k, output_dir_path, algorithm_name)
        solutions.append((solution, k))
    return solutions

def randomized_initialization_find_k(data):
    ks = [i for i in range(2, 10)]
    k = random.choice(ks)
    initial_solution = randomized_initialization(data, k)
    return initial_solution

def randomized_initialization(data, k):
    columns = data.columns
    instances = data[columns]
    scaler = StandardScaler()
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