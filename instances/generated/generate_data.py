import sys
import numpy as np
from matplotlib import pyplot as plt
import random

def read_parameters(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    sample_num = int(lines[0])
    sample_size = int(lines[1])
    mu, sigma = float(lines[2].split()[0]), float(lines[2].split()[1])
    x_min, x_max = float(lines[3].split()[0]), float(lines[3].split()[1])
    attrib_num = int(lines[4])
    a, b = [], []
    for i in range(5, sample_num+5):
        line = lines[i].strip().split()
        coefs = []
        for j in range(attrib_num):
            coefs.append(float(line[j]))
        a.append(coefs)
        b.append(float(line[-1]))
    return sample_num, sample_size, mu, sigma, x_min, x_max, attrib_num, a, b

def generate_noise(sample_size, mu, sigma):
    return np.random.randn(sample_size)*sigma + mu

def generate_linear_regression_sample(sample_size, mu, sigma, x_min, x_max, attrib_num, a, b):
    noise = generate_noise(sample_size, mu, sigma)
    # x = np.random.rand(sample_size, attrib_num)
    x = np.random.uniform(x_min, x_max, (sample_size, attrib_num))
    y = []
    for sample in range(sample_size):
        Y = 0.0
        for attrib in range(attrib_num):
            Y += a[attrib]*x[sample][attrib]
        Y += b + noise[sample]
        y.append(Y)
    return x, y

def visualize_data(X, Y, sample_num, file_path):
    plt.figure()
    for sample in range(sample_num):
        plt.scatter(X[sample], Y[sample])
        plt.xlabel('x')
        plt.ylabel('y')
    plt.savefig(file_path + '/data.png')
    plt.close()
    # plt.show()

def write_data(X, Y, sample_num, sample_size, attrib_num, file_path):
    data = []
    for sample in range(sample_num):
        sample_data = []
        for i in range(sample_size):
            line = ''
            for j in range(attrib_num):
                x = round(float(X[sample][i][j]), 2)
                line += str(x) + ' '
            y = round(Y[sample][i], 2)
            line += str(y) + '\n'     
            
            data.append(line)
            sample_data.append(line)

        with open(file_path + f'/data_sample_{sample+1}.txt', 'w') as f:
            f.writelines(sample_data)

    random.shuffle(data)

    with open(file_path + '/data.txt', 'w') as f:
        f.writelines(data)
  
def main(input_file, file_path):
    sample_num, sample_size, mu, sigma, x_min, x_max, attrib_num, a, b = read_parameters(input_file)
    X, Y = [], []
    for i in range(sample_num):
        x, y = generate_linear_regression_sample(sample_size, mu, sigma, x_min, x_max, attrib_num, a[i], b[i])
        X.append(x)
        Y.append(y)
    if attrib_num == 1:
        visualize_data(X, Y, sample_num, file_path)
    write_data(X, Y, sample_num, sample_size, attrib_num, file_path)

# python3 generate__data.py input_args_file_path output_file_dir_path
if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])