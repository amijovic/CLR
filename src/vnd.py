from copy import deepcopy
import time
import sys
import pandas as pd

from utils import print_progress, read_data_from_file, cluster_visualization, write_results
from initial_solution import initialization
from algorithm import move_one_inst, move_one_inst_approximation, move_l_instances
from algorithm import move_instances_to_one_cluster, local_search_swap

def vnd(data, k, algorithm, algorithm_name, output_dir_path, option):
    if algorithm == "move_l":
        l = 3       # okoline 3, 2, 1
        # l = 1       # okoline 1, 2, 3
        solution_not_improved = 0
        map = {3:2, 2:1, 1:3}

    if algorithm == 'move_cluster':
        cluster = 0
        first_round = True

    best_solution_labels, best_solution_regr_coefs, best_solution_regr_interception, \
        best_solution_error, best_solution_mse, best_solution_nearest_clusters = initialization(
            data, 
            k, 
            output_dir_path,
            algorithm_name
    )

    error, mse = float('inf'), float('inf')
    labels, regr_coefs, regr_interception, nearest_clusters = None, None, None, None

    first_improvement = True
    solution_improved = True
    iter = 1
    while True:
        if first_improvement == True:
            print_progress('.')
        else:
            print_progress('*')

        if first_improvement == True:
            match algorithm:
                case 'move':
                    first_improvement, error, mse, labels, nearest_clusters, regr_coefs, regr_interception = move_one_inst_approximation(
                        data,
                        best_solution_nearest_clusters,
                        best_solution_labels,
                        best_solution_error,
                        best_solution_mse,
                        best_solution_regr_coefs,
                        best_solution_regr_interception,
                        k
                    )
                case 'move_l':
                    print_progress(f'{l}')

                    first_improvement, error, mse, labels, nearest_clusters, regr_coefs, regr_interception = move_l_instances(
                        data,
                        best_solution_nearest_clusters,
                        best_solution_labels,
                        best_solution_error,
                        best_solution_mse,
                        best_solution_regr_coefs,
                        best_solution_regr_interception,
                        l,
                        k
                    )

                    if not first_improvement:
                        # l = l % 3 + 1 # okoline 1, 2, 3
                        l = map[l]      # okoline 3, 2, 1
                        solution_not_improved += 1
                    else:
                        solution_not_improved = 0

                    if solution_not_improved < 3:
                        first_improvement = True

                case 'move_cluster':
                    while first_round:
                        first_improvement, error, mse, labels, nearest_clusters, regr_coefs, regr_interception = move_instances_to_one_cluster(
                            data,
                            best_solution_nearest_clusters,
                            best_solution_labels,
                            best_solution_error,
                            best_solution_mse,
                            best_solution_regr_coefs,
                            best_solution_regr_interception,
                            cluster,
                            k
                        )
                        cluster += 1

                        if cluster == k:
                            first_round = False

                    cluster = cluster % k
                    first_improvement, error, mse, labels, nearest_clusters, regr_coefs, regr_interception = move_instances_to_one_cluster(
                        data,
                        best_solution_nearest_clusters,
                        best_solution_labels,
                        best_solution_error,
                        best_solution_mse,
                        best_solution_regr_coefs,
                        best_solution_regr_interception,
                        cluster,
                        k
                    )
                    cluster += 1
                case 'swap':
                    first_improvement, error, mse, labels, nearest_clusters, regr_coefs, regr_interception = local_search_swap(
                        data,
                        best_solution_nearest_clusters,
                        best_solution_labels,
                        best_solution_error,
                        best_solution_mse,
                        best_solution_regr_coefs,
                        best_solution_regr_interception,
                        k
                    )

            if first_improvement == True:
                best_solution_labels = deepcopy(labels)
                best_solution_nearest_clusters = deepcopy(nearest_clusters)
                best_solution_regr_coefs = deepcopy(regr_coefs)
                best_solution_regr_interception = deepcopy(regr_interception)
                best_solution_mse = mse
                best_solution_error = error
            
        else:
            solution_improved, mse, labels, nearest_clusters, regr_coefs, regr_interception = move_one_inst(
                data,
                best_solution_nearest_clusters,
                best_solution_labels,
                best_solution_mse,
                best_solution_regr_coefs,
                best_solution_regr_interception,
                k
            )

            if solution_improved == True:
                best_solution_labels = deepcopy(labels)
                best_solution_nearest_clusters = deepcopy(nearest_clusters)
                best_solution_regr_coefs = deepcopy(regr_coefs)
                best_solution_regr_interception = deepcopy(regr_interception)
                best_solution_mse = mse
                best_solution_error = error
        
        if option != 1 and first_improvement == False:
            solution_improved = False

        if not solution_improved:
            print_progress('\nsolution not improved')
            break

        if iter > 100000:
            print_progress('\nexceded iteration number')
            break
        
        iter += 1

    return best_solution_mse, best_solution_regr_coefs, best_solution_regr_interception, best_solution_labels
 
def main(input_file_path, output_dir_path, k, algorithm, option):
    time_start = time.time()

    file_data, columns = read_data_from_file(input_file_path)
    data = pd.DataFrame(file_data, columns=columns)
    algorithm_name = algorithm

    data.columns
    if option == 1:
        algorithm_name += "_1"

    best_solution_mse, best_solution_regr_coefs, best_solution_regr_interception, best_solution_labels = vnd(
        data, 
        k, 
        algorithm,
        algorithm_name,
        output_dir_path,
        option
    )

    execution_time = time.time() - time_start
    
    print_progress('\n')
    print_progress('Execution time:', str(execution_time), '\n')
    print_progress('Best solution mse:', str(best_solution_mse), '\n')
    print_progress('Best solution coefs:\n\t', str(best_solution_regr_coefs), '\n')
    print_progress('Best solution inteception:\n\t', str(best_solution_regr_interception), '\n')

    if data.shape[1] == 2:
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
    option = int(sys.argv[5])
    main(input_file_path, output_dir_path, k, algorithm, option)