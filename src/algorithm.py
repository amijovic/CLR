import random
from copy import deepcopy

from utility_functions import recalc_elastic_net, calculate_nearest_clusters, regression_error
from utility_functions import calc_error_after_change, calc_error_after_change_approximation

def move_one_inst(data, nearest_clusters, labels, best_solution_mse, best_solution_regr_coefs, best_solution_regr_interception, k):
    instance_idx = [i for i in range(data.shape[0])]
    shuffled_instances = sorted(instance_idx, key=lambda x: random.random())

    i = 0
    while i < len(shuffled_instances):
        instance = shuffled_instances[i]
        for j in nearest_clusters[instance]:
            if j == labels[instance]:
                break

            mse, regr_coefs, regr_interception, new_labels = calc_error_after_change(instance, j, data, labels, best_solution_regr_coefs, best_solution_regr_interception, k)
            if mse < best_solution_mse:
                new_nearest_clusters = calculate_nearest_clusters(data, regr_coefs, regr_interception, k)
                return True, mse, new_labels, new_nearest_clusters, regr_coefs, regr_interception
        i += 1

    return False, None, None, None, None, None

def move_one_inst_approximation(data, nearest_clusters, labels, best_solution_error, best_solution_mse, best_solution_regr_coefs, best_solution_regr_interception, k):
    instance_idx = [i for i in range(data.shape[0])]
    shuffled_instances = sorted(instance_idx, key=lambda x: random.random())

    eps = 0.01
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

            if error < (best_solution_error + eps):
                clusters = [labels[instance], j]
                new_labels = deepcopy(labels)
                new_labels[instance] = j
                regr_coefs, regr_interception = recalc_elastic_net(data, new_labels, clusters, best_solution_regr_coefs, best_solution_regr_interception)
                # new_nearest_clusters = recalc_nearest_clusters(data, nearest_clusters, regr_coefs, regr_interception, instance, k)
                new_nearest_clusters = calculate_nearest_clusters(data, regr_coefs, regr_interception, k)
                error_after_change, mse = regression_error(data, new_labels, regr_coefs, regr_interception, k)
                
                if mse < best_solution_mse:
                    return True, error_after_change, mse, new_labels, new_nearest_clusters, regr_coefs, regr_interception
        i += 1

    return False, None, None, None, None, None, None

def move_l_instances(data, nearest_clusters, labels, best_solution_error, best_solution_mse, best_solution_regr_coefs, best_solution_regr_interception, L, k):
    instance_idx = [i for i in range(data.shape[0])]
    shuffled_instances = sorted(instance_idx, key=lambda x: random.random())

    eps = 0.01
    l = 0
    l_insts  = [] # pair (inst_idx, cluster)
    error = best_solution_error
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
                error, 
                best_solution_regr_coefs, 
                best_solution_regr_interception
            )

            if error < (best_solution_error + eps):
                l += 1
                l_insts.append((instance, j))
                break

        if l == L:
            cluster_idx = set()
            new_labels = deepcopy(labels)
            for (inst, cluster) in l_insts:
                cluster_idx.add(cluster)
                cluster_idx.add(labels[inst])
                new_labels[inst] = cluster
            clusters = list(cluster_idx)
            regr_coefs, regr_interception = recalc_elastic_net(data, new_labels, clusters, best_solution_regr_coefs, best_solution_regr_interception)
            new_nearest_clusters = calculate_nearest_clusters(data, regr_coefs, regr_interception, k)
            error_after_change, mse = regression_error(data, new_labels, regr_coefs, regr_interception, k)

            if mse < best_solution_mse:
                return True, error_after_change, mse, new_labels, new_nearest_clusters, regr_coefs, regr_interception
        i += 1

    return False, None, None, None, None, None, None

def move_instances_to_one_cluster(data, nearest_clusters, labels, best_solution_error, best_solution_mse, best_solution_regr_coefs, best_solution_regr_interception, c, k):
    instance_idx = [i for i in range(data.shape[0])]
    shuffled_instances = sorted(instance_idx, key=lambda x: random.random())

    eps = 0.01
    insts  = [] # [inst_idx]
    error = best_solution_error
    i = 0
    while i < len(shuffled_instances):
        instance = shuffled_instances[i]
        
        if labels[instance] == c:
                i += 1
                continue
        
        error = calc_error_after_change_approximation(
            instance, 
            c, 
            data, 
            labels, 
            error, 
            best_solution_regr_coefs, 
            best_solution_regr_interception
        )

        if error < (best_solution_error + eps):
            insts.append(instance)
        i += 1

    if error < best_solution_error:
        clusters = [c]
        new_labels = deepcopy(labels)
        for inst in insts:
            clusters.append(labels[inst])
            new_labels[inst] = c
        regr_coefs, regr_interception = recalc_elastic_net(data, new_labels, clusters, best_solution_regr_coefs, best_solution_regr_interception)
        new_nearest_clusters = calculate_nearest_clusters(data, regr_coefs, regr_interception, k)
        error_after_change, mse = regression_error(data, new_labels, regr_coefs, regr_interception, k)

        if mse < best_solution_mse:
            return True, error_after_change, mse, new_labels, new_nearest_clusters, regr_coefs, regr_interception

    return False, None, None, None, None, None, None

def main():
    return

if __name__ == '__main__':
    main()