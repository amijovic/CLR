import os
import sys
import pandas as pd
from matplotlib import pyplot as plt
from copy import deepcopy
from sklearn.preprocessing import StandardScaler

# show_progress = True
show_progress = False

def print_progress(*msg):
    if show_progress:
        for m in msg:
            sys.stdout.write(m)
        sys.stdout.flush()

def read_data_from_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        x = line.strip().split()
        data.append([float(i) for i in x])

    m = len(data[0]) - 1
    columns = [f'x{i}' for i in range(m)]
    columns.append('y')
    
    return data, columns

def read_meteorological_predictions_data(file_path):
    df = pd.read_csv(file_path)
    df['YYYY-MM-DD'] = pd.to_datetime(df['YYYY-MM-DD'])
    df['Year-Month'] = df['YYYY-MM-DD'].dt.to_period('M')
    columns = ['max_temp', 'min_temp', 'vp', 'evap_pan', 'radiation', 'daily_rain']
    monthly_averages = df.groupby('Year-Month')[columns].mean().reset_index()
    return monthly_averages[columns], columns

def data_preprocessing(data):
    target = data.columns[-1]
    X, y = data.drop(target, axis=1), data[target]
    
    scaler = StandardScaler()
    scaler.fit(X)
    X = pd.DataFrame(scaler.transform(X), columns=data.columns[:-1])

    data_scaled = deepcopy(X)
    data_scaled[target] = y.values
    
    return data_scaled

def cluster_visualization(data, centers, labels, k, regr_coefs, regr_intercept, dir_path, algorithm_name, file_name):
    x = [min(data['x0']), max(data['x0'])]
    plt.figure()
    for c in range(k):
        isinstances = data[labels == c]
        plt.scatter(isinstances['x0'], isinstances['y'])

        y = [regr_coefs[c][0]*x[0] + regr_intercept[c], regr_coefs[c][0]*x[1] + regr_intercept[c]]
        plt.plot(x, y)

    if centers is not None:
        plt.scatter(centers['x0'], centers['y'], marker='X', color='black')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Clustered data plot')

    dir_path = os.path.join(dir_path, algorithm_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = os.path.join(dir_path, file_name)
    plt.savefig(file_path)
    plt.close()

def write_results(time, k, mse, regr_coefs, regr_interception, labels, file_path, rmse, param_values, param_scores):
    # plot_path = os.path.join(dir_path, 'train_rmse_plot.png')
    # plt.plot(param_values, param_scores)
    # plt.xlabel('k')
    # plt.ylabel('RMSE')
    # plt.savefig(plot_path)
    # plt.close()

    with open(file_path, 'w') as f:
        f.write('Execution time: ' + str(time) + '\n')
        f.write('Best model k: ' + str(k) + '\n')
        f.write('Best model train mse: ' + str(mse) + '\n')
        f.write('Best model test rmse: ' + str(rmse) + '\n')
        f.write('Best model regression coefs:\n\t' + str(regr_coefs) + '\n')
        f.write('Best model regression interception:\n\t' + str(regr_interception) + '\n')
        f.write('Clusters:\n' + str(labels) + '\n')

def save_results(execution_time, k, mse, regr_coefs, regr_interception, labels, data, file_path, rmse, param_values, param_scores, params=None):
    print_progress('\n')
    if params is not None:
        print_progress('Best model parameters: ', str(params), '\n')
    print_progress('Execution time:', str(execution_time), '\n')
    print_progress('Best model k:', str(k), '\n')
    print_progress('Best model mse:', str(mse), '\n')
    print_progress('Best model coefs:\n\t', str(regr_coefs), '\n')
    print_progress('Best model inteception:\n\t', str(regr_interception), '\n')
    print_progress('MRSE: ', str(rmse))
    
    # if data.shape[1] == 2:
    #     data = data_preprocessing(data)
    #     cluster_visualization(
    #         data, 
    #         None, 
    #         labels, 
    #         k, 
    #         regr_coefs, 
    #         regr_interception, 
    #         output_dir_path, 
    #         algorithm,
    #         'final_clustered_plot.png'
    #     )

    write_results(
        execution_time,
        k,
        mse, 
        regr_coefs, 
        regr_interception,
        labels,
        file_path,
        rmse,
        param_values,
        param_scores
    ) 

def main():
    return

if __name__ == "__main__":
    main()