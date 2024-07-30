import sys
import numpy as np
from sklearn.model_selection import train_test_split 

from utils import read_data_from_file, simple_visualization 
from kmeans import KMeans, cluster_visualization
from elastic_net import ElasticNetRegression, write_results_to_file

def main(input_file_path, output_file_path, k, kmeans_iterations, learning_rate, iterations, l1_ratio, l2_ratio):
    data = read_data_from_file(input_file_path)
    if len(data[0]) == 2:
        simple_visualization(output_file_path, '/initial_plot.png', data)

    kmeans_model = KMeans(k, kmeans_iterations)
    kmeans_model.fit(data)
    if len(data[0]) == 2:
        cluster_visualization(output_file_path, kmeans_model.clusters, kmeans_model.centers)

    clusters = kmeans_model.clusters
    centers = kmeans_model.centers
    for i in range(len(centers)):
        cluster_inst = clusters[i]
        X = np.array(cluster_inst)[:, :-1]
        y = np.array(cluster_inst)[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0) 

        en_model = ElasticNetRegression(learning_rate, iterations, l1_ratio, l2_ratio)
        en_model.fit(X_train, y_train)

        y_pred = en_model.predict(X_test)
        write_results_to_file(output_file_path, f'results_cluster_{i}', en_model.W, en_model.b, X_test, y_test, y_pred)

# python3 test.py data_file_path output_file_path k iterations learning_rate iterations l1_ratio l2_ratio
if __name__ == '__main__':
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    k = int(sys.argv[3])
    kmeans_iterations = int(sys.argv[4])
    learning_rate = float(sys.argv[5])
    iterations = int(sys.argv[6])
    l1_ratio = float(sys.argv[7])
    l2_ratio = float(sys.argv[8])
    main(input_file_path, output_file_path, k, kmeans_iterations, learning_rate, iterations, l1_ratio, l2_ratio)