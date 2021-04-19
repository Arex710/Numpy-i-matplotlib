import numpy as np

def y2(x):
    y = np.zeros(np.size(x));
    for i in range(0, np.size(x)):
        if x[i] < 0:
            y[i] = np.sin(x[i])
        else:
            y[i] = np.sqrt(x[i])
    return y
