import numpy as np


class SVM:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, iterations=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.iterations = iterations
        self.w = None
        self.b = None

    def fit(self, X, y):
        samples, features = X.shape

        new_y = np.where(y <= 0, -1, 1)

        self.w = np.zeros(features)
        self.b = 0

        for _ in range(self.iterations):
            for index, new_x in enumerate(X):
                condition = new_y[index] * (np.dot(new_x, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(new_x, new_y[index]))
                    self.b -= self.lr * new_y[index]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)
