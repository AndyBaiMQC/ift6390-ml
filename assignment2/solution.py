import numpy as np


class SVM:
    def __init__(self, eta, C, niter, batch_size, verbose=False):
        self.eta = eta
        self.C = C
        self.niter = niter
        self.batch_size = batch_size
        self.verbose = verbose

    def make_one_versus_all_labels(self, y, m):
        """
        y : numpy array of shape (n,)
        m : int (in this homework, m will be 10)
        returns : numpy array of shape (n,m)
        """
        n = y.shape[0]
        indicator_matrix = np.ones((n,m), dtype=int)
        indicator_matrix *= -1 #matrix full of -1
        for i in range(n):
            indicator_matrix[i][y[i]] = 1
        return indicator_matrix

    def compute_loss(self, x, y):
        """
        x : numpy array of shape (minibatch size, 401)
        y : numpy array of shape (minibatch size, 10)
        returns : float
        """
        n = x.shape[0]
        self.m = self.w.shape[1]
        l = self.get_matrix_l(x, y)
        l = np.power(l, 2)
        hinge_loss = self.C/n * np.sum(l)

        l2 = np.power(self.w, 2)
        l2 = np.sum(l2)/2
        return l2 + hinge_loss

    def compute_gradient(self, x, y):
        """
        x : numpy array of shape (minibatch size, 401)
        y : numpy array of shape (minibatch size, 10)
        returns : numpy array of shape (401, 10)
        """
        n = x.shape[0]
        grad_L2 = self.w
        loss_indicators = self.get_matrix_l(x,y)
        loss_indicators[loss_indicators > 0] = 1 #(m,n) where matrix[j,i] = indicator{l(w^j;(x_i,y_i))}
        w_x = np.matmul(self.w.T, x.T)  # matrix (m,n) where matrix[j,i]= <w^j,x_i>
        indicator_matrix = np.multiply(loss_indicators, (w_x - y.T)) #matrix (m,n) whre matrix[j,i] = indicator_l(w^j;(x_i,y_i)) *(<w^j,x_i> - indicator(y_i=j))
        grad_hinge_loss = np.matmul(indicator_matrix, x).T #(p,m)
        grad = grad_L2 + (self.C*2*grad_hinge_loss/n)
        return grad

    # Batcher function
    def minibatch(self, iterable1, iterable2, size=1):
        l = len(iterable1)
        n = size
        for ndx in range(0, l, n):
            index2 = min(ndx + n, l)
            yield iterable1[ndx: index2], iterable2[ndx: index2]

    def infer(self, x):
        """
        x : numpy array of shape (number of examples to infer, 401)
        returns : numpy array of shape (number of examples to infer, 10)
        """
        preds = np.dot(x, self.w)
        highest_per_row = preds.max(axis=1, keepdims=True)
        preds[:] = np.where(preds == highest_per_row, 1, -1)
        preds = preds.astype(int)
        return preds

    def compute_accuracy(self, y_inferred, y):
        """
        y_inferred : numpy array of shape (number of examples, 10)
        y : numpy array of shape (number of examples, 10)
        returns : float
        """
        class_predictions = np.argmax(y_inferred, axis=1)
        real_class = np.argmax(y, axis=1)
        #        return 1 - np.mean(y_inferred * y < 0)
        return np.sum(class_predictions == real_class) / y.shape[0]


    def fit(self, x_train, y_train, x_test, y_test):
        """
        x_train : numpy array of shape (number of training examples, 401)
        y_train : numpy array of shape (number of training examples, 10)
        x_test : numpy array of shape (number of training examples, 401)
        y_test : numpy array of shape (number of training examples, 10)
        returns : float, float, float, float
        """
        self.num_features = x_train.shape[1]
        self.m = y_train.max() + 1
        y_train = self.make_one_versus_all_labels(y_train, self.m)
        y_test = self.make_one_versus_all_labels(y_test, self.m)
        self.w = np.zeros([self.num_features, self.m])
        #self.w = np.random.randn(self.num_features, self.m) * np.sqrt(2 / (self.num_features + self.m))

        for iteration in range(self.niter):
            # Train one pass through the training set
            for x, y in self.minibatch(x_train, y_train, size=self.batch_size):
                grad = self.compute_gradient(x, y)
                self.w -= self.eta * grad

            # Measure loss and accuracy on training set
            train_loss = self.compute_loss(x_train, y_train)
            y_inferred = self.infer(x_train)
            train_accuracy = self.compute_accuracy(y_inferred, y_train)

            # Measure loss and accuracy on test set
            test_loss = self.compute_loss(x_test, y_test)
            y_inferred = self.infer(x_test)
            test_accuracy = self.compute_accuracy(y_inferred, y_test)

            if self.verbose:
                print(grad)
                print("Iteration %d:" % iteration)
                print("Train accuracy: %f" % train_accuracy)
                print("Train loss: %f" % train_loss)
                print("Test accuracy: %f" % test_accuracy)
                print("Test loss: %f" % test_loss)
                print("")

        return train_loss, train_accuracy, test_loss, test_accuracy

    def get_matrix_l(self, x, y):
        """
        x : numpy array of shape (minibatch size, 401) i.e (n,p)
        y : numpy array of shape (minibatch size, 10) i.e (n,m)
        return: numpy array of shape (10, minibatch size) ie (m,n) where matrix[j,i] =  max(0, 1 - <w^j,x_i> * indicator function{y_i=j})
        """
        self.m = self.w.shape[1]
        w_x = np.matmul(self.w.T, x.T)  # matrix (m,n) where matrix[j,i]= <w^j,x_i>
        w_x_indicator = np.multiply(w_x, self.make_one_versus_all_labels(y,self.m).T)  # matrice (m,n) where matrix[j,i] = <w^j,x_i> * indicator function{y_i=j}
        w_x_indicator = 1 - w_x_indicator  # matrice (m,n) where matrix[j,i] = 1 - <w^j,x_i> * indicator function{y_i=j}
        l = np.maximum(np.zeros((w_x_indicator.shape)),
                       w_x_indicator)  # each element, max(0, 1 - <w^j,x_i> * indicator function{y_i=j})
        return l


if __name__ == "__main__":
    # Load the data files
    print("Loading data...")
    x_train = np.load("train_features.npy")
    x_test = np.load("test_features.npy")
    y_train = np.load("train_labels.npy")
    y_test = np.load("test_labels.npy")

    print("Fitting the model...")
    svm = SVM(eta=0.001, C=30, niter=200, batch_size=5000, verbose=True)

    svm = SVM(eta=0.001, C=30, niter=200, batch_size=5000, verbose=False)
    train_loss, train_accuracy, test_loss, test_accuracy = svm.fit(x_train, y_train, x_test, y_test)

    # to infer after training, do the following:
    y_inferred = svm.infer(x_test)

    # to compute the gradient or loss before training, do the following:
    y_train_ova = svm.make_one_versus_all_labels(y_train, 10) # one-versus-all labels
    svm.w = np.zeros([401, 10])
    grad = svm.compute_gradient(x_train, y_train_ova)
    print(grad)
    loss = svm.compute_loss(x_train, y_train_ova)

    #Test gradient before fit
    svm.w = np.zeros((401, 10))

    correct = np.load('grad_before_fit_truth.npy')
    grad = svm.compute_gradient(x_train, y_train)
    print(np.allclose(correct, grad))
    for j in range(correct.shape[0]):
        for i in range(correct.shape[1]):
            if correct[j][i] != grad[j][i]:
            #if not np.allclose(correct[j][i], grad[j][i]):
                print(j,i)
                print(correct[j][i])
                print(grad[j][i])

    #train_loss, train_accuracy, test_loss, test_accuracy = svm.fit(x_train, y_train, x_test, y_test)

# # to infer after training, do the following:
# y_inferred = svm.infer(x_test)

## to compute the gradient or loss before training, do the following:
# y_train_ova = svm.make_one_versus_all_labels(y_train, 10) # one-versus-all labels
# svm.w = np.zeros([401, 10])
# grad = svm.compute_gradient(x_train, y_train_ova)
# loss = svm.compute_loss(x_train, y_train_ova)