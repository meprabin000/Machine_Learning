import numpy as np

phi = np.array([[1, 1, 1], [5.3, 7.1, 6.4]]).T
w = np.array([[-1.5, 2.4]]).T
t = np.array([[9.6, 4.2, 4.2]]).T
print(phi@w)
print(np.sum((t - phi@w)**2))
