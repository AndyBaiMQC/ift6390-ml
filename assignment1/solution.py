import numpy as np

iris = np.loadtxt('iris.txt')

######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################


class Q1:

    def feature_means(self, iris):
        self.iris = iris
        return np.transpose(np.mean(np.transpose(iris), axis=1)[0:4])
        pass

    def covariance_matrix(self, iris):
        self.iris = iris
        return np.cov(np.transpose(iris)[0:4])
        pass

    def feature_means_class_1(self, iris):
        pass

    def covariance_matrix_class_1(self, iris):
        pass


class HardParzen:
    def __init__(self, h):
        self.h = h

    def train(self, train_inputs, train_labels):
        # self.label_list = np.unique(train_labels)
        pass

    def compute_predictions(self, test_data):
        pass


class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma  = sigma

    def train(self, train_inputs, train_labels):
        # self.label_list = np.unique(train_labels)
        pass

    def compute_predictions(self, test_data):
        pass

# Problematic with AutoGrading, already emailed prof, skip for now
def split_dataset(iris):
    training_set = np.zeros(shape=(90, 5))
    validation_set = np.zeros(shape=(30, 5))
    test_set = np.zeros(shape=(30, 5))
    i = 0
    j = 0
    k = 0
    m = 0
    while i < 150:
        if i % 5 == 0 or i % 5 == 1 or i % 5 == 2:
            training_set[j] = iris[i]
            j += 1
        elif i % 5 == 3:
            validation_set[k] = iris[i]
            k += 1
        elif i % 5 == 4:
            test_set[m] = iris[i]
            m += 1
        i += 1

    result = (training_set, validation_set, test_set)

    return result

    pass


class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        pass

    def soft_parzen(self, sigma):
        pass


def get_test_errors(iris):
    pass


def random_projections(X, A):
    pass