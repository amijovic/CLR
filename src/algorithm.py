import random
from copy import deepcopy

from utility_functions import recalc_elastic_net, calculate_nearest_clusters, regression_error
from utility_functions import calc_error_after_change, calc_error_after_change_approximation, dist_to_regression_line

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
                [instance], 
                [j], 
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
                [instance], 
                [j], 
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
            else:
                l = 0
                l_insts = []
                error = best_solution_error
        i += 1

    return False, best_solution_error, best_solution_mse, labels, nearest_clusters, best_solution_regr_coefs, best_solution_regr_interception

def move_instances_to_one_cluster(data, labels, best_solution_error, best_solution_mse, best_solution_regr_coefs, best_solution_regr_interception, c, k):
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
            [instance], 
            [c], 
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
        error_after_change, mse = regression_error(data, new_labels, regr_coefs, regr_interception, k)

        if mse < best_solution_mse:
            new_nearest_clusters = calculate_nearest_clusters(data, regr_coefs, regr_interception, k)
            return True, error_after_change, mse, new_labels, new_nearest_clusters, regr_coefs, regr_interception

    return False, None, None, None, None, None, None

def local_search_swap(data, nearest_clusters, labels, best_solution_error, best_solution_mse, best_solution_regr_coefs, best_solution_regr_interception, k):
    instance_idx = [i for i in range(data.shape[0])]
    shuffled_instances = sorted(instance_idx, key=lambda x: random.random())

    eps = 0.01
    idx1 = 0
    while idx1 < len(shuffled_instances):
        instance1 = shuffled_instances[idx1]
        for cluster1 in nearest_clusters[instance1]:
            idx2 = idx1 + 1
            while idx2 < len(shuffled_instances):
                instance2 = shuffled_instances[idx2]
                cluster2 = labels[instance2]
                if cluster2 == cluster1:
                    break

                error = calc_error_after_change_approximation(
                    [instance1, instance2], 
                    [cluster2, cluster1], 
                    data, 
                    labels, 
                    best_solution_error, 
                    best_solution_regr_coefs, 
                    best_solution_regr_interception
                )

                if error < (best_solution_error + eps):
                    clusters = [labels[instance2], labels[instance1]]
                    new_labels = deepcopy(labels)
                    new_labels[instance1] = cluster2
                    new_labels[instance2] = cluster1
                    
                    regr_coefs, regr_interception = recalc_elastic_net(data, new_labels, clusters, best_solution_regr_coefs, best_solution_regr_interception)
                    new_nearest_clusters = calculate_nearest_clusters(data, regr_coefs, regr_interception, k)
                    error_after_change, mse = regression_error(data, new_labels, regr_coefs, regr_interception, k)
                    
                    if mse < best_solution_mse:
                        return True, error_after_change, mse, new_labels, new_nearest_clusters, regr_coefs, regr_interception
                idx2 += 1
        idx1 += 1

    return False, None, None, None, None, None, None

def main():
    return

if __name__ == '__main__':
    main()