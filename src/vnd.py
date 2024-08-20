import sys
import pandas as pd
import numpy as np
import random
import time
from copy import deepcopy

from sklearn.metrics import mean_squared_error

from utils import read_data_from_file, write_results
from utils import cluster_visualization, elastic_net
from initial_solution import initial_solution

def dist_to_regression_line(point, regr_coef, regr_interception):
    x, y = point[0], point[1]
    return abs(regr_coef*x - y + regr_interception) / np.sqrt(regr_coef**2 + regr_interception**2)

def calculate_nearest_clusters(data, regr_coefs, regr_interception, k):
    nearest_clusters = []
    for i in range(data.shape[0]):
        instance = data.iloc[i,].values
        sorted_clusters = [j for j in range(k)]
        sorted_clusters = sorted(
            sorted_clusters,
            key=lambda c: dist_to_regression_line(instance, regr_coefs[c], regr_interception[c])
        )
        nearest_clusters.append(sorted_clusters)
    return nearest_clusters

def predict(X, regr_coef, regr_interception):
    prediction = map(lambda x: regr_coef*x + regr_interception, X)
    return list(prediction)

def regression_error(data, labels, regr_coefs, regr_interception, k):
    Y_values = data.loc[:, 'y'].values
    Y_predictions = []
    for c in range(k):
        cluster = data[labels == c]
        X = cluster.loc[:, 'x'].values
        y_pred = predict(X, regr_coefs[c], regr_interception[c])
        Y_predictions = Y_predictions + y_pred

    error = np.sum(abs(np.subtract(Y_values, Y_predictions)))
    mse = mean_squared_error(Y_values, Y_predictions)
    return error, mse

def recalc_nearest_clusters(data, nearest_clusters, regr_coefs, regr_interception, inst_idx, k):
    new_nearest_clusters = deepcopy(nearest_clusters)
    instance = data.iloc[inst_idx,].values
    sorted_clusters = [j for j in range(k)]
    sorted_clusters = sorted(
        sorted_clusters, 
        key=lambda x: dist_to_regression_line(instance, regr_coefs[x], regr_interception[x])
    )
    new_nearest_clusters[inst_idx] = sorted_clusters
    return new_nearest_clusters

def calc_error_after_change(inst_idx, cluster_idx, data, labels, k):
    new_labels = deepcopy(labels)
    new_labels[inst_idx] = cluster_idx
    regr_coefs, regr_interception = elastic_net(data, new_labels, k)
    error, mse = regression_error(data, new_labels, regr_coefs, regr_interception, k)
    return error, mse, regr_coefs, regr_interception, new_labels

def move_one_inst(data, nearest_clusters, labels, best_solution_mse, best_solution_regr_coefs, best_solution_regr_interception):
    instance_idx = [i for i in range(data.shape[0])]
    shuffled_instances = sorted(instance_idx, key=lambda x: random.random())

    i = 0
    while i < len(shuffled_instances):
        instance = shuffled_instances[i]
        for j in nearest_clusters[instance]:
            if j == labels[instance]:
                break

            error, mse, regr_coefs, regr_interception, new_labels = calc_error_after_change(instance, j, data, labels, k)
            if mse < best_solution_mse:
                new_nearest_clusters = recalc_nearest_clusters(data, nearest_clusters, regr_coefs, regr_interception, instance, k)
                return True, error, mse, new_labels, new_nearest_clusters, regr_coefs, regr_interception
        i += 1

    return False, None, None, None, None, None, None

def calc_error_after_change_approximation(inst_idx, new_cluster_idx, data, labels, error, regr_coefs, regr_interception):
    point = data.iloc[inst_idx]
    x, y = point['x'], point['y']
    old_cluster_idx = labels[inst_idx]
    new_error = error - abs(y - (regr_coefs[old_cluster_idx]*x + regr_interception[old_cluster_idx]))
    new_error = new_error + abs(y - (regr_coefs[new_cluster_idx]*x + regr_interception[new_cluster_idx]))
    return new_error

def move_one_inst_approximation(data, nearest_clusters, labels, best_solution_error, best_solution_regr_coefs, best_solution_regr_interception):
    instance_idx = [i for i in range(data.shape[0])]
    shuffled_instances = sorted(instance_idx, key=lambda x: random.random())

    i = 0
    while i < len(shuffled_instances):
        instance = shuffled_instances[i]
        for j in nearest_clusters[instance]:
            if j == labels[instance]:
                break

            error = calc_error_after_change_approximation(
                instance, 
                j, 
                data, 
                labels, 
                best_solution_error, 
                best_solution_regr_coefs, 
                best_solution_regr_interception
            )

            if error < best_solution_error:
                new_labels = deepcopy(labels)
                new_labels[instance] = j
                regr_coefs, regr_interception = elastic_net(data, new_labels, k)
                new_nearest_clusters = recalc_nearest_clusters(data, nearest_clusters, regr_coefs, regr_interception, instance, k)
                error_after_change, mse = regression_error(data, labels, regr_coefs, regr_interception, k)
                return True, error_after_change, mse, new_labels, new_nearest_clusters, regr_coefs, regr_interception
        i += 1

    return False, None, None, None, None, None, None

def main(input_file_path, output_dir_path, k, algorithm_name):
    time_start = time.time()
    data = pd.DataFrame(read_data_from_file(input_file_path), columns=['x', 'y'])
    
    best_solution_labels, best_solution_regr_coefs, best_solution_regr_interception = initial_solution(
        data, 
        k, 
        output_dir_path,
        algorithm_name
    )
    best_solution_error, best_solution_mse = regression_error(
        data, 
        best_solution_labels,
        best_solution_regr_coefs, 
        best_solution_regr_interception,
        k
    )
    best_solution_nearest_clusters = calculate_nearest_clusters(
        data, 
        best_solution_regr_coefs, 
        best_solution_regr_interception,
        k
    )

    iter = 1
    while True:
        sys.stdout.write('.')
        sys.stdout.flush()
        solution_improved, error, mse, labels, nearest_clusters, regr_coefs, regr_interception = move_one_inst_approximation(
            data,
            best_solution_nearest_clusters,
            best_solution_labels,
            best_solution_error,
            best_solution_regr_coefs,
            best_solution_regr_interception
        )

        # solution_improved, error, mse, labels, nearest_clusters, regr_coefs, regr_interception = move_one_inst(
        #     data,
        #     best_solution_nearest_clusters,
        #     best_solution_labels,
        #     best_solution_mse,
        #     best_solution_regr_coefs,
        #     best_solution_regr_interception
        # )

        if not solution_improved:
            print('\nsolution not improved')
            break

        if iter > 100000:
            print('\nexceded iteration number')
            break
        
        best_solution_labels = labels
        best_solution_nearest_clusters = nearest_clusters
        best_solution_regr_coefs = regr_coefs
        best_solution_regr_interception = regr_interception
        best_solution_mse = mse
        best_solution_error = error
        iter += 1

    execution_time = time.time() - time_start
    
    print()
    print('Execution time:', execution_time)
    print('Best solution mse:', best_solution_mse)
    print('Best solution coefs:\n\t', best_solution_regr_coefs)
    print('Best solution inteception:\n\t', best_solution_regr_interception)
    cluster_visualization(
        data, 
        None, 
        best_solution_labels, 
        k, 
        best_solution_regr_coefs, 
        best_solution_regr_interception, 
        output_dir_path, 
        algorithm_name,
        'final_clustered_plot.png'
    )
    write_results(
        execution_time,
        best_solution_mse, 
        best_solution_regr_coefs, 
        best_solution_regr_interception,
        best_solution_labels,
        output_dir_path,
        algorithm_name
    )

if __name__ == '__main__':
    input_file_path = sys.argv[1]
    output_dir_path = sys.argv[2]
    k = int(sys.argv[3])
    algorithm = sys.argv[4]

    algorithm_name = []
    match algorithm:
        case 'move':
            algorithm_name = 'move_one_inst'
    main(input_file_path, output_dir_path, k, algorithm_name)