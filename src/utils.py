import os
from matplotlib import pyplot as plt

def read_data_from_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        x = line.strip().split()
        data.append([float(i) for i in x])

    return data

def cluster_visualization(data, centers, labels, k, regr_coefs, regr_intercept, dir_path, algorithm_name, file_name):
    x = [min(data['x']), max(data['x'])]
    plt.figure()
    for c in range(k):
        isinstances = data[labels == c]
        plt.scatter(isinstances['x'], isinstances['y'])

        y = [regr_coefs[c][0]*x[0] + regr_intercept[c], regr_coefs[c][0]*x[1] + regr_intercept[c]]
        plt.plot(x, y)

    if centers is not None:
        plt.scatter(centers['x'], centers['y'], marker='X', color='black')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Clustered data plot')
    # plt.legend()
    file_path = os.path.join(dir_path, algorithm_name, file_name)
    plt.savefig(file_path)

def write_results(time, mse, regr_coefs, regr_interception, labels, dir_path, algorithm_name):
    file_path = os.path.join(dir_path, algorithm_name, 'results.txt')
    with open(file_path, 'w') as f:
        f.write('Execution time: ' + str(time) + '\n')
        f.write('mse: ' + str(mse) + '\n')
        f.write('Regression coefs:\n\t' + str(regr_coefs) + '\n')
        f.write('Regression interception:\n\t' + str(regr_interception) + '\n')
        f.write('Clusters:\n' + str(labels) + '\n')

def main():
    return

if __name__ == "__main__":
    main()