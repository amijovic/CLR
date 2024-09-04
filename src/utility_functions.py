import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from copy import deepcopy

def elastic_net(data, labels, k):
    regr_coefs = []
    regr_intercept = []
    for c in range(k):
        cluster = data[labels == c]
        model = ElasticNet(random_state=0)
        y = cluster.loc[:, 'y'].values
        X = cluster.loc[:, 'x'].values.reshape((y.shape[0], 1))
        model.fit(X, y)
        regr_coefs.append(model.coef_)
        regr_intercept.append(model.intercept_)
    return regr_coefs, regr_intercept

def recalc_elastic_net(data, labels, clusters, regr_coefs, regr_intercept):
    new_regr_coefs = deepcopy(regr_coefs)
    new_regr_intercept = deepcopy(regr_intercept)
    for c in clusters:
        cluster = data[labels == c]
        model = ElasticNet(random_state=0)
        y = cluster.loc[:, 'y'].values
        X = cluster.loc[:, 'x'].values.reshape((y.shape[0], 1))
        model.fit(X, y)
        new_regr_coefs[c] = model.coef_
        new_regr_intercept[c] = model.intercept_    
    return new_regr_coefs, new_regr_intercept

def dist_to_regression_line(point, regr_coef, regr_interception):
    x, y = point[0], point[1]
    return abs(regr_coef*x - y + regr_interception) / np.sqrt(regr_coef**2 + (-1)**2)

def calculate_nearest_clusters(data, regr_coefs, regr_interception, k):
    nearest_clusters = []
    for i in range(data.shape[0]):
        instance = data.iloc[i,].values
        sorted_clusters = [j for j in range(k)]
        sorted_clusters = sorted(
            sorted_clusters,
            key=lambda c: dist_to_regression_line(instance, regr_coefs[c], regr_interception[c])
        )
        nearest_clusters.append(sorted_clusters)
    return nearest_clusters

def predict(X, regr_coef, regr_interception):
    prediction = map(lambda x: regr_coef*x + regr_interception, X)
    return list(prediction)

def regression_error(data, labels, regr_coefs, regr_interception, k):
    Y_values = []
    Y_predictions = []
    for c in range(k):
        cluster = data[labels == c]
        X = cluster.loc[:, 'x'].values
        Y = cluster.loc[:, 'y'].values
        y_pred = predict(X, regr_coefs[c], regr_interception[c])
        Y_predictions = Y_predictions + y_pred
        Y_values = Y_values + list(Y)

    error = np.sum(abs(np.subtract(Y_values, Y_predictions)))
    mse = mean_squared_error(Y_values, Y_predictions)
    return error, mse

def regression_mse(data, labels, regr_coefs, regr_interception, k):
    Y_values = []
    Y_predictions = []
    for c in range(k):
        cluster = data[labels == c]
        X = cluster.loc[:, 'x'].values
        Y = cluster.loc[:, 'y'].values
        y_pred = predict(X, regr_coefs[c], regr_interception[c])
        Y_predictions = Y_predictions + y_pred
        Y_values = Y_values + list(Y)

    mse = mean_squared_error(Y_values, Y_predictions)
    return mse

def calc_error_after_change(inst_idx, cluster_idx, data, labels, regr_coefs, regr_interception, k):
    new_labels = deepcopy(labels)
    new_labels[inst_idx] = cluster_idx
    clusters = [cluster_idx, labels[inst_idx]]
    new_regr_coefs, new_regr_interception = recalc_elastic_net(data, new_labels, clusters, regr_coefs, regr_interception)
    mse = regression_mse(data, new_labels, new_regr_coefs, new_regr_interception, k)
    return mse, new_regr_coefs, new_regr_interception, new_labels

def calc_error_after_change_approximation(inst_idx, new_cluster_idx, data, labels, error, regr_coefs, regr_interception):
    point = data.iloc[inst_idx]
    x, y = point['x'], point['y']
    old_cluster_idx = labels[inst_idx]
    new_error = error - abs(y - (regr_coefs[old_cluster_idx]*x + regr_interception[old_cluster_idx]))
    new_error = new_error + abs(y - (regr_coefs[new_cluster_idx]*x + regr_interception[new_cluster_idx]))
    return new_error

def main():
    return

if __name__ == "__main__":
    main()