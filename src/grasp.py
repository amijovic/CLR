import sys
import pandas as pd
import time
from copy import deepcopy

from utils import print_progress, read_data_from_file, data_preprocessing, save_results
from initial_solution import randomized_initialization
from algorithm import local_search_swap

def grasp(data, k):    
    best_solution_labels, best_solution_regr_coefs, best_solution_regr_interception = None, None, None 
    best_solution_mse = float('inf')
    best_solution_params = ''

    n = 15
    for i in range(n):
        sys.stdout.write('-')
        sys.stdout.flush()

        labels, regr_coefs, regr_interception, error, mse, nearest_clusters, params = randomized_initialization(data, k)
        
        data_scaled = data_preprocessing(data)

        solution_improved = True
        iter = 1
        while True:
            solution_improved, new_error, new_mse, new_labels, new_nearest_clusters, new_regr_coefs, new_regr_interception = local_search_swap(
                data_scaled,
                nearest_clusters,
                labels,
                error,
                mse,
                regr_coefs,
                regr_interception,
                k
            )

            if not solution_improved:
                print_progress('\nsolution not improved\n')
                break

            if iter > 100000:
                print_progress('\nexceded iteration number\n')
                break

            labels = deepcopy(new_labels)
            nearest_clusters = deepcopy(new_nearest_clusters)
            error = new_error
            mse = new_mse
            regr_coefs = deepcopy(new_regr_coefs)
            regr_interception = deepcopy(new_regr_interception)
            iter += 1
        
        if mse < best_solution_mse:
            best_solution_mse = mse
            best_solution_labels = deepcopy(labels)
            best_solution_regr_coefs = deepcopy(regr_coefs)
            best_solution_regr_interception = deepcopy(regr_interception)
            best_solution_params = params

    return best_solution_mse, best_solution_regr_coefs, best_solution_regr_interception, best_solution_labels, best_solution_params

def main(input_file_path, output_dir_path, k, algorithm):
    time_start = time.time()

    file_data, columns = read_data_from_file(input_file_path)
    data = pd.DataFrame(file_data, columns=columns)
 
    best_solution_mse, best_solution_regr_coefs, best_solution_regr_interception, best_solution_labels, best_solution_params = grasp(data, k)
   
    execution_time = time.time() - time_start
    save_results(
        execution_time,
        k,
        best_solution_mse,
        best_solution_regr_coefs,
        best_solution_regr_interception,
        best_solution_labels,
        data,
        algorithm,
        output_dir_path,
        best_solution_params
    )

if __name__ == '__main__':
    input_file_path = sys.argv[1]
    output_dir_path = sys.argv[2]
    k = int(sys.argv[3])
    algorithm = "grasp"
    main(input_file_path, output_dir_path, k, algorithm)