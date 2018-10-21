#############################################
## author: siwa (juierror)                 ##
## last modify: 21/10/2018                 ##
#############################################

import numpy as np
from GuassianNB import GuassianNB

# data from https://en.wikipedia.org/wiki/Naive_Bayes_classifier example sex classification

# featues height,weigth,foot size
x = np.array(
    [
        [6, 180, 12],
        [5.92, 190, 11],
        [5.58, 170, 12],
        [5.92, 165, 10],
        [5, 100, 6],
        [5.5, 150, 8],
        [5.42, 130, 7],
        [5.75, 150, 9],
    ]
)

# class 0 => male , 1 => female
y = np.array([0, 0, 0, 0, 1, 1, 1, 1]).reshape(-1, 1)

# predict female
x_test = np.array([[6, 130, 8]])

model = GuassianNB(x, y)
model.train()
pred = model.test(x_test)
print(pred)

