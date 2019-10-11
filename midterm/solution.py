import numpy as np

def ones_at_the_end(x):
    """
    :param x: python int list
    :return: python int list
    """
    ones = [_x for _x in x if _x == 1]
    not_ones = [_x for _x in x if _x != 1]
    new_list = not_ones[::-1] + ones
    return new_list

# %%
def final_position(instructions):
    """
    :param instructions: string
    :return: int tuple
    """
    pos = [0, 0]
    instructions = instructions.split()
    for ins in instructions:
        if ins == 'left':
            pos[0] -= 1
        elif ins == 'right':
            pos[0] += 1
        elif ins == 'up':
            pos[1] += 1
        elif ins == 'down':
            pos[1] -= 1
            
    return tuple(pos)

# %%
def steps_to_one(i):
    """
    :param i: int
    :return:  int
    """
    count = 0
    while i != 1:
        if i % 2 == 0:
            i = i / 2
        else:
            i = 3 *i + 1
        count += 1
        
    return count

def find_bins(input_list, k):
    """
    :param input_list: list of k*h floats
    :param k: int
    :return: list of k+1 floats
    """
    if input_list == None:
        return None
    
    input_list = sorted(input_list)
    bins = [input_list[0] - .5]
    h = len(input_list) / k
    for i in range(0, len(input_list)-1):
        if (i+1) % h == 0:
            bins.append(.5*input_list[i]+.5*input_list[i+1])
    bins.append(input_list[-1] + .5)

    return bins

def even_odd_ordered(X):
    """
    :param X: np.array of shape (n,)
    :return: np.array of shape (n,)
    """
    return np.array([_x for _x in X if _x%2 == 0] 
                    + [_x for _x in X if _x%2 != 0])

def data_normalization(X):
    """
    :param X: np.array of shape n x (d+1)
    :return: np.array of shape n x (d+1)
    """
    x_mean = np.mean(X[:,:-1], axis=0)
    x_std = np.std(X[:,:-1], axis=0)

    for i in range(X.shape[1]-1):
        X[:,i] = (X[:,i] - x_mean[i]) / x_std[i]

    return X

# %%
def entropy(p):
    """
    :param p: np.array of shape (n,)
    :return: float or None
    """
    if np.abs(np.sum(p) - 1.) > 1e-10: # should be near to 1
        return None
    
    if np.sum(p < 0) > 0: # should be non-negative
        return None
    
    ent = 0.
    for _p in p:
        if np.abs(_p) > 1e-10:
            ent -= _p * np.log2(_p)
            
    return ent

def heavyball_optimizer(x, inputs, alpha=0.9, beta=0.1):

    x_old = np.zeros(len(x))
    for t in range(len(inputs)):
        x_new = x - alpha * inputs[t] + beta * (x - x_old)
        x_old = x
        x = x_new
        
    return x

# %%
class NearestCentroidClassifier:
    def __init__(self, k, d):
        self.k = k
        self.centroids = np.zeros((k, d))

    def fit(self, X, y):  # question A
        for k in range(self.k):
            self.centroids[k] = np.mean(X[y == k, :])
            
    def predict(self, X):  # question B
        dist = np.zeros((X.shape[0], self.k))
        for i in range(X.shape[0]):
            dist[i,:] = np.sum((X[i,:] - self.centroids)**2, axis=1)
        return np.argmin(dist, axis=1)

    def score(self, X, y):  # question C
        pred = self.predict(X)
        return np.mean(pred == y)


# %%
def test_centroid_classifer():
    train_points = np.array([[0.], [1.], [5.], [4.], [4.], [4.]])
    train_labels = np.array([0, 0, 0, 1, 1, 1])

    test_points = np.array([[0.], [1.], [2.], [3.], [4.], [5.], [0.]])
    test_labels = np.array([0, 0, 0, 0, 1, 1, 1])

    k = 2
    d = 1

    clf = NearestCentroidClassifier(k, d)
    clf.fit(train_points, train_labels)
    predictions = clf.predict(test_points)
    score = clf.score(test_points, test_labels)
    print(f'Your classifier predicted {predictions}')
    print(f'This gives it a score of {score}')


if __name__ == '__main__':
    test_centroid_classifer()