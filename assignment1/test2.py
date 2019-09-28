import numpy as np

iris = np.loadtxt('iris.txt')

arr = np.argsort(iris[:, 4])
result = np.zeros(shape=(150, 5))
i = 0
for num in arr:
    if i < 150:
        result[i] = iris[num]
    i = i + 1
x = np.split(result,3)
# y = np.transpose(x)
# z = np.mean(y, axis=1)
# np.transpose(np.mean(np.transpose(x), axis=1)[0:4])
# 5.01, 3.43, 1.46, 0.25
print(x[0])
print(round(155/3))
