import numpy as np

def make_array_from_list(some_list):
    return np.array(some_list)
    pass

def make_array_from_number(num):
    return np.array([num]*1)
    pass

class NumpyBasics:
    def add_arrays(self, a, b):
        self.a = a
        self.b = b
        return np.add(self.a, self.b)
        pass

    def add_array_number(self, a, num):
        self.a = a
        self.num = num
        return np.add(self.a, [self.num]*len(a))
        pass

    def multiply_elementwise_arrays(self, a, b):
        self.a = a
        self.b = b
        return np.multiply(self.a, self.b)
        pass

    def dot_product_arrays(self, a, b):
        self.a = a
        self.b = b
        sum = 0
        for x, y in zip(self.a, self.b):
            sum += np.multiply(x, y)
        return sum
        pass

    def dot_1d_array_2d_array(self, a, m):
        self.a = a
        self.m = m
        return np.dot(self.a, self.m)
        # consider the 2d array to be like a matrix
        pass