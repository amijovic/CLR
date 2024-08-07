import sys
from matplotlib import pyplot as plt

def read_data_from_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        x = line.strip().split()
        data.append([float(i) for i in x])

    return data

def simple_visualization(data, file_path, file_name):
    X = data['x']
    y = data['y']
    plt.scatter(X, y)
    plt.xlabel('x')
    plt.xlabel('y')
    plt.title('Initial data plot')
    plt.savefig(file_path + '/' + file_name)

def cluster_visualization(data, centers, labels, k, file_path, file_name):
    plt.figure()
    for c in range(k):
        isinstances = data[labels == c]
        plt.scatter(isinstances['x'], isinstances['y'])
    if centers is not None:
        plt.scatter(centers['x'], centers['y'], marker='X', color='black')
    plt.xlabel('x')
    plt.xlabel('y')
    plt.title('Clustered data plot')
    # plt.legend()
    plt.savefig(file_path + '/' + file_name)

def main():
    return

if __name__ == "__main__":
    main()