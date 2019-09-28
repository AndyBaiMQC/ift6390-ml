import numpy as np

iris = np.loadtxt('iris.txt')

train_set = iris[[ i for i in range(0, 150) if i % 5 == 0 or i % 5 == 1 or i % 5 == 2]]
valid_set = iris[[ i for i in range(0, 150) if i % 5 == 3]]
test_set = iris[[ i for i in range(0, 150) if i % 5 == 4]]

result = (train_set, valid_set, test_set)
print(result)

a = [4, 3, 5, 7, 6, 8]

tr = []
va = []
tst = []
i = 0

while i < 150:
    if i % 5 == 0 or i % 5 == 1 or i % 5 == 2:
        tr.append(i)
    elif i % 5 == 3:
        va.append(i)
    elif i % 5 == 4:
        tst.append(i)
    i += 1

train = np.take(iris[tr])
print(train)