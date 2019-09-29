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

    # hidden test...
    def feature_means_class_1(self, iris):
        self.iris = iris
        result = np.zeros(shape=(np.shape(iris)))
        i = 0
        for num in np.argsort(iris[:, 4]):
            if i < len(iris):
                result[i] = iris[num]
            i = i + 1
        return np.transpose(np.mean(np.transpose(result[0:round(len(iris)/3)]), axis=1)[0:4])
        pass

    def covariance_matrix_class_1(self, iris):
        self.iris = iris
        result = np.zeros(shape=(np.shape(iris)))
        i = 0
        for num in np.argsort(iris[:, 4]):
            if i < len(iris):
                result[i] = iris[num]
            i = i + 1
        return np.cov(np.transpose(result[0:round(len(iris)/3)])[0:4])
        pass
    
    def feature_means_class_1(self, iris):
        self.iris = iris
        
        # cal the mask of is_class_1
        is_class_1 = iris[:, -1] == 1
        
        # select data of class 1
        class_1_feature = iris[is_class_1, :-1]
        
        return class_1_feature.mean(axis=0)

    def covariance_matrix_class_1(self, iris):
        self.iris = iris
        
        # cal the mask of is_class_1
        is_class_1 = iris[:, -1] == 1
        
        # select data of class 1
        class_1_feature = iris[is_class_1, :-1]
        
        return np.cov(class_1_feature.T)

class HardParzen:
    def __init__(self, h):
        self.h = h

    def train(self, train_inputs, train_labels):
        self.label_list = label_list = np.unique(train_labels)
        self.train_data = train_inputs
        self.train_label = train_labels
        
        return
    
    
    
    def is_in_the_window(self, center, inputs):
        def cal_euclidean_distance(x, y):
#             print(np.sqrt(np.sum((x-y)**2, axis=-1)))
            return np.sqrt(np.sum((x-y)**2, axis=-1))
        
        # use euclidean distance as metric
        return cal_euclidean_distance(center, inputs) < self.h
    

    def compute_predictions(self, test_data):
        predictions = []
        for te in test_data:
            # find neighbors of te
            is_neighbors = self.is_in_the_window(te, self.train_data)
            
            if is_neighbors.sum() == 0:
                predictions.append(int(draw_rand_label(te, self.label_list)))
                continue
            
            # get the label list of its neighbors
            neighbor_labels = self.train_label[is_neighbors]
            
            
            # cal the counts of each class
            uniqued, counts = np.unique(neighbor_labels, return_counts=True)
            
            # the prediction is the lable with largest count
            predictions.append(int(uniqued[counts.argmax()]))
        
        return np.array(predictions)
    
    


class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma  = sigma

    def train(self, train_inputs, train_labels):
        self.label_list = label_list = np.unique(train_labels)
        self.train_data = train_inputs
        self.train_label = train_labels
        self.dim = self.train_data.shape[-1]
        
        return

    def compute_predictions(self, test_data):
        predictions = []
        
        def cal_euclidean_distance(x, y):
            return np.sqrt(np.sum((x-y)**2, axis=-1))
        
        # cal kernel density estimates
        def cal_kde(x, mean, sigma):
#             former = 1. / ( (2*np.pi)**(mean.shape[-1]/2) * np.linalg.det(sigma)**0.5 )
#             latter = np.exp( -0.5 * (x-mean).dot(np.linalg.inv(sigma)).dot((x-mean).T) )
            former = 1. / ( (2*np.pi)**(self.dim/2) * sigma**self.dim )
            latter = np.exp( -0.5 * cal_euclidean_distance(x, mean)**2 / sigma**2 )
            
            return former * latter
        
        for te in test_data:
            weights = []
            for l in self.label_list:
                # select all data of class l
                data_l = self.train_data[self.train_label==l]
                
                # cal the weight for all data_l
                weight_l = cal_kde(data_l, mean=te, sigma=self.sigma)
                
                # add the total weight of class l to weights
                weights.append(weight_l.sum())
                
            # the current prediction is the label with largest weight
            predictions.append(int(self.label_list[np.argmax(weights)]))
        
        assert predictions is not None
        return np.array(predictions)
    

def split_dataset(iris):
    training_set = np.zeros(shape=(90, 5))
    validation_set = np.zeros(shape=(30, 5))
    test_set = np.zeros(shape=(30, 5))
    i = 0
    j = 0
    k = 0
    m = 0
    while i < len(iris):
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
    
    def cal_error_rate(self, pred, label):
        return (pred.astype(int)!=label.astype(int)).sum() / label.shape[0]
    
    def hard_parzen(self, h):
        hp = HardParzen(h=h)

        hp.train(self.x_train, self.y_train)

        pred = hp.compute_predictions(self.x_val)
        
        return self.cal_error_rate(pred, self.y_val)

    def soft_parzen(self, sigma):
        sp = SoftRBFParzen(sigma=sigma)

        sp.train(self.x_train, self.y_train)

        pred = sp.compute_predictions(self.x_val)
        
        return self.cal_error_rate(pred, self.y_val)


def get_test_errors(iris):
    hs = [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 15.0, 20.0]
    sigmas = [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 15.0, 20.0]
    
    (training_set, validation_set, test_set) = split_dataset(iris)
    
    tr_x, tr_y = training_set[:, :-1], training_set[:, -1]
    val_x, val_y = validation_set[:, :-1], validation_set[:, -1]
    te_x, te_y = test_set[:, :-1], test_set[:, -1]
    
    er = ErrorRate(tr_x, tr_y, val_x, val_y)
    
    er_hp = np.array([er.hard_parzen(h) for h in hs])
    er_sp = np.array([er.soft_parzen(sigma) for sigma in sigmas])
    
    
    h_star = hs[er_hp.argmin()]
    sigma_star = sigmas[er_sp.argmin()]
    
    er_te = ErrorRate(tr_x, tr_y, te_x, te_y)
    
    
    
#     if (er_sp.argmin()==er_sp).sum()>1:
        
#         soft_results = []
#         for ss in sigmas[er_sp==er_sp.argmin()]:
#             soft_results.append(er_te.soft_parzen(ss))
            
#         return np.array([er_te.hard_parzen(h_star), min(soft_results)])
#     print(h_star, sigma_star,er_hp, er_sp, h_star, sigma_star)
    
    
    
    results = np.array([er_te.hard_parzen(h_star), er_te.soft_parzen(sigma_star)])
    
    return results
        
    
def random_projections(X, A):
    return 1./2**0.5 * A.T.dot(X.T).T