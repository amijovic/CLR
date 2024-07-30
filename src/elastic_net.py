import numpy as np   
import pandas as pd 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 
  
class ElasticNetRegression() : 
    def __init__(self, learning_rate, iterations, l1_ratio, l2_ratio) : 
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio

    def fit(self, X, Y):
        self.W = np.zeros(X.shape[1])
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.iterations):
            self.update_weights()
        return self
      
    def update_weights(self):
        Y_pred = self.predict(self.X)
        
        m, n = self.X.shape
        dW = np.zeros(n)
        for j in range(n):               
            w_sgn = 1
            if self.W[j] < 0:
                w_sgn = -1
        dW[j] = (-2*(self.X[:, j]).dot(self.Y - Y_pred)
                 + w_sgn*self.l1_ratio + 2*self.l2_ratio*self.W[j]) / m 
        db = -2*np.sum(self.Y - Y_pred) / m  

        self.W = self.W - self.learning_rate*dW 
        self.b = self.b - self.learning_rate*db 
        return self

    def predict(self, X) :       
        return X.dot(self.W) + self.b 
          
def write_results_to_file(file_path, file_name, weights, b, X_test, y_test, y_pred):
    with open(file_path + '/' + file_name + '.txt', 'w') as f:
        f.write("Trained W: ")
        f.write(str(round(weights[0], 2)))
        f.write('\n')
        f.write("Trained b: ")
        f.write(str(round(b, 2)))

    plt.figure()
    plt.scatter(X_test, y_test, color = 'blue')    
    plt.plot(X_test, y_pred, color = 'red')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.savefig(file_path + '/' + file_name + '.png')
    
def main() : 
    return
          
if __name__ == "__main__" :  
    main()