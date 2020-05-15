import dataset_manipulation as utils
import numpy as np
import pandas as pd

class LogisticRegression:
    import numpy as np
    
    def __init__(self, max_iter=1000, epsilon=0.1, kernel=None):
        self.theta = None
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.fitted = False
        self.kernel = kernel
    
    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    def _kernel(self, x):
        if self.kernel == 'ellipsoid':
            return self.theta.dot(x**2)
        if self.kernel == 'paraboloid':
            return self.theta[0]*x[0] + self.theta[1:].dot(x[1:])
        return self.theta.dot(x)
    
    def _hypothesis(self, x):
        return self._sigmoid(self._kernel(x))
    
    def _add_ones(self, X):
        ones = np.ones((X.shape[0],1))
        return np.concatenate([X, ones], axis=1)
    
    def _randomize_theta(self, X):
        self.theta = np.random.uniform(low=-3, high=3, size=X.shape[1]) # assumes bias (aka _add_ones) was added
    
    def _gradient(self, X, target):
        n = X.shape[0] # num of datapoints
        d = self.theta.shape[0] # dimension of data vector space
        gradient_J = np.zeros(d)
        for j in range(d):
            for i in range(n):
                gradient_J[j] += target[i] * X[i][j] - X[i][j] * self._hypothesis(X[i])
        return gradient_J
    
    def _gradient_descent(self, X, target, lr= 0.1, max_iter=1000):
        for i in range(max_iter):
            new_theta = self.theta + lr * self._gradient(X, target)
            if (sum(abs(new_theta - self.theta) > self.epsilon)) == 0: # number of components greater than epsilon
                break
            self.theta = new_theta
    
    def fit(self, X, target):
        X = self._add_ones(X) # add bias vector
        self._randomize_theta(X)
        self._gradient_descent(X, target)
        self.fitted = True
    
    def predict_proba(self, X):
        X = self._add_ones(X)
        probabilities = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            probabilities[i] = self._hypothesis(X[i])
        return probabilities
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
        
    
    
if __name__ == "__main__":
    
    #utils.build_dataset()
    dataset = pd.read_csv("toy_dataset.csv")
    utils.visualize_dataset(dataset)
    
    model = LogisticRegression(kernel='ellipsoid')
    X      = dataset.iloc[:,:-1].values
    target = dataset.target.values
    
    utils.visualize_decision_boundary(dataset, lambda x,y: model.theta[0]*x**2 + model.theta[1]*y**2 + model.theta[2])
