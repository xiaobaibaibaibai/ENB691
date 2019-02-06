import sys
import numpy as np

print(sys.version_info)
print(np.__version__)
print ('Hello CSE691!')
'''
# In Python, it uses line indentation to indicate a block of code 
if True:
    print ('True')
else:
    print ('False')

# Quotation: You can use either single quote or double quote for string.
word = 'Hello'
sentence = "Hi, how are you?"
print (word)
print (sentence)

# Comment
print # I'm a comment

# Similar to any other languages, but you don't need to specify whether it is integer or float. 
x = 10
print (type(x))
print (x)
print (x + 1)
print (x - 1)
print (x * 2)
print (x ** 2)
x += 1
print (x)
x *= 2
print (x)
y = 2.5
print (type(y))
print (y, y + 1, y * 2, y ** 2)
print (x + y)
print (x > y)

# String: Single quote or Double quote.
str1 = 'Hello'
str2 = "CSE"
print (str1)
print(len(str1))
concat = str1 + ' ' + str2
print (concat)
formatStr = '{0} {1} {2}'.format(str1, str2, 691)
print (formatStr)

# Booleans: Use word instead of symbol (||, &&)
t = True
f = False
print (type(t))
print (t and f)
print (t or f)
print (not t)

# It is similar to array, but can store different type of elements in the same array
numList = [0] * 5
print(numList)
numList[0] = 7
numList[4] = 1
print(numList)
numList[2] = 'Hello'
numList.append('World')
print (numList)
print (numList[-1])

# Number list
numList = list(range(10))
print(numList)
print(numList[2:4])
print(numList[2:])
print(numList[:-1])
print(numList[:])
numList[2:5] = [1, 0, 1]
print(numList)

# For loop
categories = [0, 'cat', 'dog']
for tmp in categories:
    print(tmp)
for x in range(10):
    print (x*2)

# List comprehension
lists = [0, 1, 2, 3, 4]
powers = [x ** 2 for x in lists]
print(powers)
powerOdd = [x ** 2 for x in lists if x % 2 != 0]
print(powerOdd)

# Dictionary
tmpDict = {'A': 30, 'B': 40, 'C': 50}
for name, age in tmpDict.items():
    print('{0} is {1} years old.'.format(name, age))

# Function
def checkNum(x):
    if (x % 2) == 0:
        return 'even'
    else:
        return "odd"
for x in range(10):
    print(checkNum(x))

# Create array
a = np.arange(15)
print(a)
print(a.shape)
b = np.zeros(10, dtype='int')
print(b)
c = np.ones((2,5), dtype='float')
print(c)
print(c.shape)
tmp = np.array(np.arange(15)).reshape((3,5))
print(tmp)
print(tmp.shape)

# Create with range that have even space.
evenSpace = np.linspace(0, 10, 5)
print(evenSpace)

# Create array with random integer array
integer = np.random.randint(10, size=(3,4,5))
print(integer.ndim)
print(integer.shape)
print(integer.size)
print(integer)

# Access, similar to access list in pure Python.
numList = np.array(range(10))
print(numList)
print(numList[2:4])
print(numList[2:])
print(numList[:-1])
print(numList[:])
numList[2:5] = [1, 0, 1]
print(numList)

# Concatenate array
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
z = [0, 7, 0]
tmp = np.concatenate([x, y, z])
print(tmp)
x = np.array([1, 2, 3])
y = np.array([[4, 5, 6], [7, 8, 9]])
z = np.vstack([y, x])
print(x.shape)
print(y.shape)
print(z)
q = np.array([[0], [17]])
r = np.hstack([y, q])
print(q.shape)
print(r)

# NumPy operation
a = np.array([20, 30, 40, 50])
b = np.arange(4)
print(a-b)
print(b**2)
print(100*np.sin(a))
print(a<37)

a = np.array([[1, 2],
              [3, 4]])
b = np.array([[2, 0],
              [3, 4]])
print(a*b)
print(a.dot(b))
print(np.dot(a, b))
'''
# Broadcasting
x = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])
y = np.array([1, 0, 1])
print(x.shape, y.shape)
z = x + y
print(z)

x = np.array([1, 2, 3])
y = np.array([4, 5])
print(np.reshape(x, (3, 1)) * y)
z = np.array([[1, 2, 3],
              [4, 5, 6]])

print((z.T + y).T)
print(x + np.reshape(y, (2, 1)))
print(x * 2)

'''
'''






