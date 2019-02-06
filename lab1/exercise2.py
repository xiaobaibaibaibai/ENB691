list = [0, 1, 2, 3, 4]

powerOdd = [x ** 2 for x in list if x % 2 != 0]

# print(powerOdd)

def isPrime(x):
    for i in range(2, x):
        if (x % i is 0):
            return False
    else: 
        return True

primeNumber = [x for x in range(2, 101) if x % 2 != 0 and isPrime(x)]
primeNumber.insert(0, 2)

print(primeNumber)