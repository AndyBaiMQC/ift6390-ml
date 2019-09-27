import numpy as np

iris = np.loadtxt('iris.txt')

training = []
validation = []
test = []

print(len(iris))
for index in (0, 150):
    if index % 5 == 0 or index % 5 == 1 or index % 5 == 2:
        np.concatenate((training, (iris[index])), axis=0) #training
    elif index % 5 == 3:
        np.concatenate((validation, (iris[index])), axis=0) #validation
    else:
        np.concatenate((test, (iris[index])), axis=0) #test

print("debug done")

