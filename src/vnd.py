import sys
import pandas as pd
import time
from copy import deepcopy

from utils import read_data_from_file, write_results, cluster_visualization
from initial_solution import initialization
from algorithm import move_one_inst_approximation, move_l_instances, move_instances_to_one_cluster, move_one_inst

def vnd1(data, k, algorithm, algorithm_name):
    best_solution_labels, best_solution_regr_coefs, best_solution_regr_interception, \
    best_solution_error, best_solution_mse, best_solution_nearest_clusters = initialization(
        data, 
        k, 
        output_dir_path,
        algorithm_name
    )

    if algorithm == 'move_cluster':
        cluster = 0
        first_round = True

    first_improvement = True
    solution_improved = True
    iter = 1
    while True:
        if first_improvement == True:
            sys.stdout.write('.')
        else:
            sys.stdout.write('*')
        sys.stdout.flush()

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
                    first_improvement, error, mse, labels, nearest_clusters, regr_coefs, regr_interception = move_l_instances(
                        data,
                        best_solution_nearest_clusters,
                        best_solution_labels,
                        best_solution_error,
                        best_solution_mse,
                        best_solution_regr_coefs,
                        best_solution_regr_interception,
                        5,
                        k
                    )
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

        if not solution_improved:
            print('\nsolution not improved')
            break

        if iter > 100000:
            print('\nexceded iteration number')
            break
        
        iter += 1

    return best_solution_mse, best_solution_regr_coefs, best_solution_regr_interception, best_solution_labels
    
def vnd2(data, k, algorithm, algorithm_name):
    best_solution_labels, best_solution_regr_coefs, best_solution_regr_interception, \
    best_solution_error, best_solution_mse, best_solution_nearest_clusters = initialization(
        data, 
        k, 
        output_dir_path,
        algorithm_name
    )

    if algorithm == 'move_cluster':
        cluster = 0
        first_round = True

    iter = 1
    while True:
        sys.stdout.write('.')
        sys.stdout.flush()

        match algorithm:
            case 'move':
                solution_improved, error, mse, labels, nearest_clusters, regr_coefs, regr_interception = move_one_inst_approximation(
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
                solution_improved, error, mse, labels, nearest_clusters, regr_coefs, regr_interception = move_l_instances(
                    data,
                    best_solution_nearest_clusters,
                    best_solution_labels,
                    best_solution_error,
                    best_solution_mse,
                    best_solution_regr_coefs,
                    best_solution_regr_interception,
                    5,
                    k
                )
            case 'move_cluster':
                while first_round:
                    solution_improved, error, mse, labels, nearest_clusters, regr_coefs, regr_interception = move_instances_to_one_cluster(
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
                solution_improved, error, mse, labels, nearest_clusters, regr_coefs, regr_interception = move_instances_to_one_cluster(
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
            
        if not solution_improved:
            print('\nsolution not improved')
            break

        if iter > 100000:
            print('\nexceded iteration number')
            break
        
        best_solution_labels = deepcopy(labels)
        best_solution_nearest_clusters = deepcopy(nearest_clusters)
        best_solution_regr_coefs = deepcopy(regr_coefs)
        best_solution_regr_interception = deepcopy(regr_interception)
        best_solution_mse = mse
        best_solution_error = error
        iter += 1

    return best_solution_mse, best_solution_regr_coefs, best_solution_regr_interception, best_solution_labels
    

def main(input_file_path, output_dir_path, k, algorithm, option):
    algorithm_name = []
    match algorithm:
        case 'move':
            algorithm_name = 'move_one_inst'
        case 'move_l':
            algorithm_name = 'move_l_inst'
        case 'move_cluster':
            algorithm_name = 'move_to cluster'

    time_start = time.time()
    data = pd.DataFrame(read_data_from_file(input_file_path), columns=['x', 'y'])
    
    if option == 1:
        best_solution_mse, best_solution_regr_coefs, best_solution_regr_interception, best_solution_labels = vnd1(
            data, 
            k, 
            algorithm, 
            algorithm_name
        )
    else:
        best_solution_mse, best_solution_regr_coefs, best_solution_regr_interception, best_solution_labels = vnd2(
            data, 
            k, 
            algorithm, 
            algorithm_name
        )

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
    option = int(sys.argv[5])

    main(input_file_path, output_dir_path, k, algorithm, option)