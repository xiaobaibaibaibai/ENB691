import numpy as np
import time
import random

# - Fill in the code below the comment Python and NumPy same as in example.
# - Follow instructions in document.
###################################################################
# Example: Create a zeros vector of size 10 and store variable tmp.
'''
# Python
pythonStartTime = time.time()
tmp_1 = [0 for i in range(10)]
pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))

# NumPy
numPyStartTime = time.time()
tmp_2 = np.zeros(10)
numPyEndTime = time.time()
print('NumPy  time: {0} sec.'.format(numPyEndTime-numPyStartTime))
'''

z_1 = None
z_2 = None
################################################################
# 1. Create a zeros array of size (3,5) and store in variable z.
# Python
print("# 1")
pythonStartTime = time.time()
z_1 = [[0]*5]*3
pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
# NumPy
numPyStartTime = time.time()
z_2 = np.zeros((3, 5), dtype= np.int)
numPyEndTime = time.time()
print('NumPy  time: {0} sec.\n'.format(numPyEndTime-numPyStartTime))

#################################################
# 2. Set all the elements in first row of z to 7.
# Python
print("# 2")
pythonStartTime = time.time()
z_1[0] = [7]*len(z_1[0])
pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
# NumPy
numPyStartTime = time.time()
z_2[:1,] = [7, 7, 7, 7, 7]
numPyEndTime = time.time()
print('NumPy  time: {0} sec.\n'.format(numPyEndTime-numPyStartTime))

#####################################################
# 3. Set all the elements in second column of z to 9.
# Python
print("# 3")
pythonStartTime = time.time()
for i in range(len(z_1)):
    z_1[i][1] = 9
pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
# NumPy
numPyStartTime = time.time()
z_2[:,1:2] = 9
numPyEndTime = time.time()
print('NumPy  time: {0} sec.\n'.format(numPyEndTime-numPyStartTime))

#############################################################
# 4. Set the element at (second row, third column) of z to 5.
# Python
print("# 4")
pythonStartTime = time.time()
z_1[1][2] = 5
pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
# NumPy
numPyStartTime = time.time()
z_2[1, 2] = 5
numPyEndTime = time.time()
print('NumPy  time: {0} sec.\n'.format(numPyEndTime-numPyStartTime))

##############
print(z_1)
print(z_2)
##############


x_1 = None
x_2 = None
##########################################################################################
# 5. Create a vector of size 50 with values ranging from 50 to 99 and store in variable x.
# Python
print("\n# 5")
pythonStartTime = time.time()
x_1 = []
for i in range(50, 100):
    x_1.append(i)
pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
# NumPy
numPyStartTime = time.time()
x_2 = np.arange(50, 100)
numPyEndTime = time.time()
print('NumPy  time: {0} sec.\n'.format(numPyEndTime-numPyStartTime))

##############
print(x_1)
print(x_2)
##############


y_1 = None
y_2 = None
##################################################################################
# 6. Create a 4x4 matrix with values ranging from 0 to 15 and store in variable y.
# Python
print("\n# 6")
pythonStartTime = time.time()
y_1 = []
for i in range(4):
    temp = []
    for j in range(4):
        temp.append(i*4+j)
    y_1.append(temp)
pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
# NumPy
numPyStartTime = time.time()
y_2 = np.arange(16).reshape(4,4)
numPyEndTime = time.time()
print('NumPy  time: {0} sec.\n'.format(numPyEndTime-numPyStartTime))

##############
print(y_1)
print(y_2)
##############


tmp_1 = None
tmp_2 = None
####################################################################################
# 7. Create a 5x5 array with 1 on the border and 0 inside and store in variable tmp.
# Python
print("\n# 7")
pythonStartTime = time.time()
tmp_1 = []
tmp_1.append([1]*5)
for i in range(1, 4):
    temp = []
    for j in range(0, 5):
        if j is 0 or j is 4:
            temp.append(1)
        else:
            temp.append(0)
    tmp_1.append(temp)
tmp_1.append([1]*5)
pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))

# NumPy
numPyStartTime = time.time()
tmp_2 = np.ones((5, 5), dtype=np.int)
tmp_2[1:4,1:4] = np.zeros(3, dtype=np.int)
numPyEndTime = time.time()
print('NumPy  time: {0} sec.\n'.format(numPyEndTime-numPyStartTime))

##############
print(tmp_1)
print(tmp_2)
##############


a_1 = None; a_2 = None
b_1 = None; b_2 = None
c_1 = None; c_2 = None
#############################################################################################
# 8. Generate a 50x100 array of integer between 0 and 5,000 and store in variable a.
# Python
print("\n# 8")
pythonStartTime = time.time()
a_1 = []
for i in range(50):
    temp = []
    for j in range(100):
        temp.append(i*100+j)
    a_1.append(temp)
pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
# NumPy
numPyStartTime = time.time()
a_2 = np.matrix(np.arange(5000, dtype=np.int).reshape(50,100))
numPyEndTime = time.time()
print('NumPy  time: {0} sec.\n'.format(numPyEndTime-numPyStartTime))

###############################################################################################
# 9. Generate a 100x200 array of integer between 0 and 20,000 and store in variable b.
# Python
print("# 9")
pythonStartTime = time.time()
b_1 = []
for i in range(100):
    temp = []
    for j in range(200):
        temp.append(i*200+j)
    b_1.append(temp)
pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
# NumPy
numPyStartTime = time.time()
b_2 = np.matrix(np.arange(20000, dtype=np.int).reshape(100,200))
numPyEndTime = time.time()
print('NumPy  time: {0} sec.\n'.format(numPyEndTime-numPyStartTime))

#####################################################################################
# 10. Multiply matrix a and b together (real matrix product) and store to variable c.
# Python
print("# 10")
pythonStartTime = time.time()
c_1 = []
for i in range(len(a_1)):
    temp = []
    for j in range(len(b_1[0])):
        num = 0
        for m in range(len(b_1)):
            num = a_1[i][m] * b_1[m][j] + num
        temp.append(num)
    c_1.append(temp)
pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
# NumPy
numPyStartTime = time.time()
c_2 = np.matmul(a_2, b_2)
numPyEndTime = time.time()
print('NumPy  time: {0} sec.\n'.format(numPyEndTime-numPyStartTime))


d_1 = None; d_2 = None
################################################################################
# 11. Normalize a 3x3 random matrix ((x-min)/(max-min)) and store to variable d.
# Python
print("# 11")
pythonStartTime = time.time()
d_1 = []
for i in range(3):
    temp = []
    for j in range(3):
        temp.append(random.random())
    d_1.append(temp)
d_1_min = 1
for i in d_1:
    for j in i:
        if j < d_1_min:
            d_1_min = j
d_1_max = 0
for i in d_1:
    for j in i:
        if j > d_1_max:
            d_1_max = j
for i in range(3):
    for j in range(3):
        d_1[i][j] = (d_1[i][j]-d_1_min)/(d_1_max-d_1_min)
pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
# NumPy
numPyStartTime = time.time()
d_2 = np.random.rand(3, 3)
d_2_min = d_2.min()
d_2_max = d_2.max()
d_2 = (d_2 - d_2_min) / (d_2_max - d_2_min)
numPyEndTime = time.time()
print('NumPy  time: {0} sec.\n'.format(numPyEndTime-numPyStartTime))
##########
print(d_1)
print(d_2)
#########


################################################
# 12. Subtract the mean of each row of matrix a.
# Python
print("\n# 12")
pythonStartTime = time.time()
for i in range(50):
    mean_row = sum(a_1[i])/100
    for j in range(100):
        a_1[i][j] = a_1[i][j] - mean_row
pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
# NumPy
numPyStartTime = time.time()
a_2 = a_2 - a_2.mean(axis=1)
numPyEndTime = time.time()
print('NumPy  time: {0} sec.\n'.format(numPyEndTime-numPyStartTime))

###################################################
# 13. Subtract the mean of each column of matrix b.
# Python
print("# 13")
pythonStartTime = time.time()
for j in range(200):
    col_sum = 0
    for i in range(100):
        col_sum += b_1[i][j]
    mean_col = col_sum / 100
    for i in range(100):
        b_1[i][j] = b_1[i][j] - mean_col
pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
# NumPy
numPyStartTime = time.time()
b_2 = b_2 - b_2.mean(axis=0)
numPyEndTime = time.time()
print('NumPy  time: {0} sec.\n'.format(numPyEndTime-numPyStartTime))

################
print(np.sum(a_1 == a_2))
print(np.sum(b_1 == b_2))
################

e_1 = None; e_2 = None
###################################################################################
# 14. Transpose matrix c, add 5 to all elements in matrix, and store to variable e.
# Python
print("\n# 14")
pythonStartTime = time.time()
e_1 = []
for i in range(len(c_1[0])):
    temp = []
    for j in range(len(c_1)):
        temp.append(c_1[j][i])
    e_1.append(temp)
pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
# NumPy
numPyStartTime = time.time()
e_2 = c_2.transpose()
numPyEndTime = time.time()
print('NumPy  time: {0} sec.\n'.format(numPyEndTime-numPyStartTime))
##################
print (np.sum(e_1 == e_2))
##################


#####################################################################################
# 15. Reshape matrix e to 1d array, store to variable f, and print shape of f matrix.
# Python
print("\n# 15")
pythonStartTime = time.time()
f_1 = []
for i in range(len(e_1)):
    for j in range(len(e_1[0])):
        f_1.append(e_1[i][j])
pythonEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
# NumPy
numPyStartTime = time.time()
f_2 = np.ravel(e_2)
numPyEndTime = time.time()
print('NumPy  time: {0} sec.\n'.format(numPyEndTime-numPyStartTime))

print("f_1 shape is ({0},)".format(len(f_1)))
print("f_2 shape is {0}".format(f_2.shape))
