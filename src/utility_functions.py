import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from copy import deepcopy

def elastic_net(data, labels, k):
    regr_coefs = []
    regr_intercept = []
    target = data.columns[-1]
    attribs = data.columns[:-1]
    m = len(attribs)

    for c in range(k):
        cluster = data[labels == c]
        model = ElasticNet(random_state=0)
        y = cluster.loc[:, target].values
        # X = cluster.loc[:, attribs].values.reshape((y.shape[0], 1))
        X = cluster.loc[:, attribs].values.reshape((y.shape[0], m))
        model.fit(X, y)
        regr_coefs.append(model.coef_)
        regr_intercept.append(model.intercept_)
    return regr_coefs, regr_intercept

def recalc_elastic_net(data, labels, clusters, regr_coefs, regr_intercept):
    new_regr_coefs = deepcopy(regr_coefs)
    new_regr_intercept = deepcopy(regr_intercept)
    target = data.columns[-1]
    attribs = data.columns[:-1]
    m = len(attribs)

    for c in clusters:
        cluster = data[labels == c]
        model = ElasticNet(random_state=0)
        y = cluster.loc[:, target].values
        X = cluster.loc[:, attribs].values.reshape((y.shape[0], m))
        model.fit(X, y)
        new_regr_coefs[c] = model.coef_
        new_regr_intercept[c] = model.intercept_    
    return new_regr_coefs, new_regr_intercept

def dist_to_regression_line(point, regr_coef, regr_interception):
    x, y = point[:-1], point[-1]
    return abs(np.dot(regr_coef, x) - y + regr_interception) / np.sqrt(np.sum(regr_coef**2) + (-1)**2)

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
    prediction = map(lambda x: np.dot(regr_coef, x) + regr_interception, X)
    return list(prediction)

def regression_error(data, labels, regr_coefs, regr_interception, k):
    Y_values = []
    Y_predictions = []
    target = data.columns[-1]
    attribs = data.columns[:-1]

    for c in range(k):
        cluster = data[labels == c]
        X = cluster.loc[:, attribs].values
        Y = cluster.loc[:, target].values
        y_pred = predict(X, regr_coefs[c], regr_interception[c])
        Y_predictions = Y_predictions + y_pred
        Y_values = Y_values + list(Y)

    error = np.sum(abs(np.subtract(Y_values, Y_predictions)))
    mse = mean_squared_error(Y_values, Y_predictions)
    return error, mse

def regression_mse(data, labels, regr_coefs, regr_interception, k):
    Y_values = []
    Y_predictions = []
    target = data.columns[-1]
    attribs = data.columns[:-1]

    for c in range(k):
        cluster = data[labels == c]
        X = cluster.loc[:, attribs].values
        Y = cluster.loc[:, target].values
        y_pred = predict(X, regr_coefs[c], regr_interception[c])
        Y_predictions = Y_predictions + y_pred
        Y_values = Y_values + list(Y)

    mse = mean_squared_error(Y_values, Y_predictions)
    return mse

def calc_error_after_change(inst_idxs, cluster_idxs, data, labels, regr_coefs, regr_interception, k):
    new_labels = deepcopy(labels)
    changed_clusters = set()
    for i, inst_idx in enumerate(inst_idxs):
        new_labels[inst_idx] = cluster_idxs[i]
        changed_clusters.add(cluster_idxs[i])
        changed_clusters.add(labels[inst_idx])
        # clusters = [cluster_idx, labels[inst_idx]]
    new_regr_coefs, new_regr_interception = recalc_elastic_net(data, new_labels, changed_clusters, regr_coefs, regr_interception)
    mse = regression_mse(data, new_labels, new_regr_coefs, new_regr_interception, k)
    return mse, new_regr_coefs, new_regr_interception, new_labels

def calc_error_after_change_approximation(inst_idxs, new_cluster_idxs, data, labels, error, regr_coefs, regr_interception):
    new_error = error
    target = data.columns[-1]
    attribs = data.columns[:-1]

    for i, inst_idx in enumerate(inst_idxs):
        new_cluster_idx = new_cluster_idxs[i]
        x, y = data.loc[inst_idx, attribs], data.loc[inst_idx, target]
        old_cluster_idx = labels[inst_idx]
        new_error = new_error - abs(y - (np.dot(regr_coefs[old_cluster_idx], x) + regr_interception[old_cluster_idx]))
        new_error = new_error + abs(y - (np.dot(regr_coefs[new_cluster_idx], x) + regr_interception[new_cluster_idx]))
    return new_error

def calc_cluster_centroids(X, k, labels):
    centroids = []
    for c in range(k):
        cluster = X[labels == c]
        centroid = np.mean(cluster)
        centroids.append(centroid)
    return centroids

def main():
    return

if __name__ == "__main__":
    main()