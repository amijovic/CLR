import sys
import pandas as pd
import numpy as np
from copy import deepcopy

from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

from utils import read_data_from_file, simple_visualization, cluster_visualization
from initial_solution import initial_solution

def dist_to_regression_line(point, regr_coef, regr_interception):
    x, y = point[0], point[1]
    return abs(regr_coef*x - y + regr_interception) / np.sqrt(regr_coef**2 + regr_interception**2)

def calculate_nearest_centers(data, regr_coefs, regr_interception, k):
    nearest_centers = []
    for i in range(data.shape[0]):
        instance = data.iloc[i,].values
        sorted_centers = [j for j in range(k)]
        sorted_centers = sorted(
            sorted_centers,
            key=lambda x: dist_to_regression_line(instance, regr_coefs[x], regr_interception[x])
        )
        nearest_centers.append(sorted_centers)
    return nearest_centers

def elastic_net(data, labels, k):
    regr_coefs = []
    regr_intercept = []
    for c in range(k):
        cluster = data[labels == c]
        regr = ElasticNet(random_state=0)
        y = cluster.loc[:, 'y'].values
        X = cluster.loc[:, 'x'].values.reshape((y.shape[0], 1))
        regr.fit(X, y)
        regr_coefs.append(regr.coef_)
        regr_intercept.append(regr.intercept_)
    return regr_coefs, regr_intercept

def predict(X, regr_coef, regr_interception):
    return regr_coef*X + regr_interception

def regression_error(data, labels, regr_coefs, regr_interception, k):
    total_error = 0
    for c in range(k):
        cluster = data[labels == c]
        X, y = cluster.loc[:, 'x'].values, cluster.loc[:, 'y'].values
        y_pred = predict(X, regr_coefs[c], regr_interception[c])
        total_error += mean_squared_error(y, y_pred)
    return total_error

def calc_error_after_change(inst_idx, cluster_idx, data, labels, k):
    new_labels = deepcopy(labels)
    new_labels[inst_idx] = cluster_idx
    regr_coefs, regr_interception = elastic_net(data, new_labels, k)
    error = regression_error(data, new_labels, regr_coefs, regr_interception, k)
    return error, regr_coefs, regr_interception, new_labels

def recalc_nearest_centers(data, nearest_centers, regr_coefs, regr_interception, inst_idx, k):
    new_nearest_centers = deepcopy(nearest_centers)
    instance = data.iloc[inst_idx,].values
    sorted_centers = [j for j in range(k)]
    sorted_centers = sorted(
        sorted_centers, 
        key=lambda x: dist_to_regression_line(instance, regr_coefs[x], regr_interception[x])
    )
    new_nearest_centers[inst_idx] = sorted_centers
    return new_nearest_centers

def local_search_swap(data, nearest_centers, labels, best_solution_error): 
    for i in range(data.shape[0]):
        for j in nearest_centers[i]:
            if j == labels[i]:
                continue

            error, regr_coefs, regr_interception, new_labels = calc_error_after_change(i, j, data, labels, k)
            if error < best_solution_error:
                new_nearest_centers = recalc_nearest_centers(data, nearest_centers, regr_coefs, regr_interception, i, k)
                return True, error, new_labels, new_nearest_centers, regr_coefs, regr_interception
    
    return False, None, None, None, None, None

def main(input_file_path, output_file_path, k):
    data = pd.DataFrame(read_data_from_file(input_file_path), columns=['x', 'y'])
    simple_visualization(data, output_file_path, 'initial_plot.png')
    best_solution_labels, best_solution_regr_coefs, best_solution_regr_interception = initial_solution(
        data, 
        k, 
        output_file_path
    )
    best_solution_error = regression_error(
        data, 
        best_solution_labels,
        best_solution_regr_coefs, 
        best_solution_regr_interception,
        k
    )
    best_solution_nearest_centers = calculate_nearest_centers(
        data, 
        best_solution_regr_coefs, 
        best_solution_regr_interception,
        k
    )

    iter = 1
    while True:
        sys.stdout.write('.')
        sys.stdout.flush()
        solution_improved, error, labels, nearest_centers, regr_coefs, regr_interception = local_search_swap(
            data,
            best_solution_nearest_centers,
            best_solution_labels,
            best_solution_error
        )

        if not solution_improved or iter > 100:
            break
        
        best_solution_labels = labels
        best_solution_nearest_centers = nearest_centers
        best_solution_regr_coefs = regr_coefs
        best_solution_regr_interception = regr_interception
        best_solution_error = error
        iter += 1

    print()
    print(best_solution_error)
    print(best_solution_regr_coefs)
    print(best_solution_regr_interception)
    cluster_visualization(data, None, best_solution_labels, k, output_file_path, 'final_clustered_plot.png')

if __name__ == '__main__':
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    k = int(sys.argv[3])
    main(input_file_path, output_file_path, k)