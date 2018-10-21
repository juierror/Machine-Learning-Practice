#############################################
## author: siwa (juierror)                 ##
## last modify: 21/10/2018                 ##
#############################################

import numpy as np


class GuassianNB:
    def __init__(self, x, y):
        # x shape (lines,features)
        self.x = x
        self.y = y
        self.c = np.unique(y)
        self.mean = dict()
        self.var = dict()
        # self.prior = dict()

    def train(self):
        for c in list(self.c):
            x_c = self.x[self.y.reshape(-1) == c]
            # find mean in each class
            mean = np.mean(x_c, axis=0)
            # find variance in each class
            var = np.var(x_c, axis=0)
            # prior = x_c.shape[0] / self.x.shape[0]
            self.mean[c] = mean
            self.var[c] = var
            # self.prior[c] = prior

    def test(self, x_test):
        # tmp for keep mul(p(Xi|Ck)) size (lines,num_class) to find argmax(mul(p(Xi|Ck)))
        tmp = np.ones((x_test.shape[0], self.c.shape[0]))
        i = 0
        for c in list(self.c):
            # find p(Xi|Ck) in each feature(Xi)
            gd = self.guassianDis(x_test, self.mean[c], self.var[c])
            for j in range(gd.shape[1]):
                # find mul(p(Xi|Ck))
                tmp[:, i] *= gd[:, j]
            # tmp[:,i] *= self.prior
            i += 1
        # find argmax(mul(p(Xi|Ck)))
        index = np.argmax(tmp, axis=1).reshape(-1)
        return np.array([self.c[e] for e in index])

    def guassianDis(self, x, mean, var):
        e = np.exp(-((x - mean) ** 2) / (2 * var))
        tmp = 1 / ((2 * np.pi * var) ** (0.5))
        return tmp * e
