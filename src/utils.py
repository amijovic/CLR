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

def simple_visualization(file_path, file_name, data):
    X = []
    y = []
    for i in range(len(data)):
        X.append(data[i][0])
        y.append(data[i][1])
    plt.scatter(X, y)
    plt.savefig(file_path + file_name)


def main():
    return

if __name__ == "__main__":
    main()