import sys
import pandas as pd
import numpy as np
import time
from copy import deepcopy
from math import sqrt

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import mean_squared_error

from utils import read_data_from_file, read_meteorological_predictions_data 
from utils import save_results
from estimator import Estimator

def load_data(input_file_path):
    # file_data, columns = read_data_from_file(input_file_path)
    file_data, columns = read_meteorological_predictions_data(input_file_path)
    data = pd.DataFrame(file_data, columns=columns)
    target = columns[-1]
    X, y = data.drop(target, axis=1), data[target]
    return X, y, columns[:-1], target

def main(input_file_path, output_dir_path, algorithm, option):
    time_start = time.time()

    X, y, attrib_columns, target = load_data(input_file_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    algorithm_name = algorithm
    if option is not None:
        if option == 1:
            algorithm_name += "_1"
        else:
            algorithm_name += "_2"
    else:
        algorithm_name = 'grasp'

    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    param_values = range(2, 10)
    best_param = None
    best_score = float('inf')
    param_scores = []
    for param in param_values:
        model = Estimator(param, algorithm_name, attrib_columns, target)
        scores = cross_validate(model, X_train.values, y_train.values, cv=kf, scoring='neg_root_mean_squared_error')
        mean_score = -np.mean(scores['test_score'])
        param_scores.append(mean_score)
        print(mean_score)

        if mean_score < best_score:
            best_score = mean_score
            best_param = param

    print(f"Best parameter: {best_param} with mean score: {best_score}")

    best_model = Estimator(best_param, algorithm_name, attrib_columns, target)
    best_model.fit(X_train.values, y_train.values)
    y_pred = best_model.predict(X_test.values)
    rmse = sqrt(mean_squared_error(y_test.values, y_pred))

    print(f'RMSE = {rmse}')

    execution_time = time.time() - time_start

    data_train = deepcopy(X_train)
    data_train[target] = y_train.values

    save_results(
        execution_time,
        best_model.k,
        best_model.mse,
        best_model.regr_coefs,
        best_model.regr_interception,
        best_model.labels,
        data_train,
        algorithm,
        output_dir_path, 
        rmse,
        param_values,
        param_scores
    )

if __name__ == '__main__':
    input_file_path = sys.argv[1]
    output_dir_path = sys.argv[2]
    algorithm = sys.argv[3]
    if len(sys.argv) == 5:
        option = int(sys.argv[4])
    else:
        option = None
    
    main(input_file_path, output_dir_path, algorithm, option)