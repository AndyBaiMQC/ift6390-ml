import numpy as np

iris = np.loadtxt('iris.txt')

result = tuple()
training = np.zeros(shape=(90, 5))
validation = np.zeros(shape=(30, 5))
test = np.zeros(shape=(30, 5))
i = 0
j = 0
k = 0
m = 0
while i < 150:
    if i % 5 == 0 or i % 5 == 1 or i % 5 == 2:
        training[j] = iris[i]
        j += 1
    elif i % 5 == 3:
        validation[k] = iris[i]
        k += 1
    elif i % 5 == 4:
        test[m] = iris[i]
        m += 1
    i += 1
print((training, validation, test))
print("--------------------------------")

train_set = iris[[ i for i in range(0, 150) if i % 5 == 0 or i % 5 == 1 or i % 5 == 2]]
valid_set = iris[[ i for i in range(0, 150) if i % 5 == 3]]
test_set = iris[[ i for i in range(0, 150) if i % 5 == 4]]

result = (train_set, valid_set, test_set)
print(result)
print(type(result))
print("end")
print(type([0, 1, 4]))

