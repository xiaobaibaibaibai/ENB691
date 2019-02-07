import numpy as np


a = np.array([
    [0, 1, 3, 4],
    [10, 31, 53, 0],
    [8, 12, 36, 0]
])
b = np.array([
    [4, 3, 2, 1],
    [4, 0, 2, 1],
    [1, 2, 2, 1],
    [1, 2, 3, 4],
])

c = np.matmul(a, b)
# print(c)


a_1 = [
    [0, 1, 3, 4],
    [10, 31, 53, 0],
    [8, 12, 36, 0]
]

b_1 = [
    [4, 3, 2, 1],
    [4, 0, 2, 1],
    [1, 2, 2, 1],
    [1, 2, 3, 4],
]

c_1 = []

for i in range(len(a_1)):
    temp = []
    for j in range(len(b_1[0])):
        num = 0
        for m in range(len(b_1)):
            num = a_1[i][m] * b_1[m][j] + num
        temp.append(num)
    c_1.append(temp)
# print(c_1)


e_1 = []
for i in range(len(c_1[0])):
    temp = []
    for j in range(len(c_1)):
        temp.append(c_1[j][i])
    e_1.append(temp)
# print(ct)

e_2 = c.transpose()



print(np.sum(e_2 == e_1))

f_2 = np.reshape(e_2, e_2.size)
print(f_2.size)
