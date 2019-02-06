import numpy as np

# Given the Celsius temperature, convert it to Fahrenheit (celsius * 1.8 = fahrenheit - 32)
# then check where the temperature in Fahrenheit is more than 75, 
# If yes print "Too hot",  otherwise print Celsius and Fahrenheit temperature on the screen.

def CelToFah(c):
    f = c * 1.8 + 32
    if (f > 75):
        print("Too hot!")
    else:
        print("Celsius: {0}, Fahrenheit {1}".format(c, f))

CelToFah(50)


# Exercise2: Find the list of prime number from 0-100. 
# Hint: Use List comprehension with function.

def isPrime(x):
    for i in range(2, x):
        if (x % i is 0):
            return False
    else: 
        return True

primeNumber = [x for x in range(2, 101) if x % 2 != 0 and isPrime(x)]
primeNumber.insert(0, 2)

# print(primeNumber)


# Exercise3: Create a 10x10 array with 0 on the border and 1 inside.


arr = np.zeros((10,10), dtype='int')
arr[1:9,1:9] = [1, 1, 1, 1, 1, 1, 1, 1]
# print(arr)
