import numpy as np

x = np.array([
    [1, 1, 2, 1],
    [2, 1, 3, 1]
])

w = np.array([
    [2, 1],
    [1, 2],
    [1, 5],
    [0.5, 2]
])

y = np.array([
    [1],
    [1]
])

S = x.dot(w)
#Sy = S[np.arange(N), y].reshape(-1, 1)
Sy = np.array([
    [5.5],
    [21]
])

margine = S - Sy + 1

print(margine)
