#############################################
## author: siwa (juierror)                 ##
## last modify: 20/10/2018                 ##
#############################################

import numpy as np


class LinearRegressor:
    def __init__(self, x, y, lr):
        # x shape (line,features)
        # y shape (line,1)
        # thata shape (features,1)
        self.x = x
        self.y = y
        self.lr = lr
        self.theta = np.random.rand(x.shape[1], 1)
        self.const = 0
        self.num = x.shape[0]

    def update(self):
        #### update theta
        # tmp shape (line,1)
        tmp = self.predict(self.x) - self.y
        # gd shape (features,1)
        gd = np.dot(self.x.T, tmp) / self.num
        self.theta = self.theta - self.lr * gd

        ### update const
        tmp = np.sum(self.predict(self.x) - self.y) / self.num
        self.const = self.const - self.lr * tmp

    def loss_func(self):
        tmp = self.y - self.predict(self.x)
        tmp = np.dot(tmp.T, tmp)
        tmp = tmp[0][0] / self.num
        return tmp

    def train(self, iter):
        for i in range(iter):
            self.update()

    def predict(self, x_test):
        # x_test shape (line,features)
        pred = np.dot(x_test, self.theta) + self.const
        return pred

