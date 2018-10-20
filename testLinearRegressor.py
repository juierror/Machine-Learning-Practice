#############################################
## author: siwa (juierror)                 ##
## last modify: 20/10/2018                 ##
#############################################

import numpy as np
from matplotlib import pyplot as plt
from LinearRegressor import LinearRegressor

x = np.array([1, 3, 2, 8, 6, 7, -3, -2, -7])
_x = x.reshape(-1, 1)
y = np.array([2, 3, 2, 8, 6, 7, -3, -4, -7])
_y = y.reshape(-1, 1)

model = LinearRegressor(_x, _y, 0.01)
model.train(30)
pred = model.predict(np.array(range(-10, 10)).reshape(-1, 1))

plt.scatter(x, y)
plt.plot(np.array(range(-10, 10)), pred.flatten())
plt.show()

