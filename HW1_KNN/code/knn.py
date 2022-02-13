import numpy as np
from scipy.spatial.distance import cdist
from scipy import stats
from statistics import mode


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, k=1):
        self.k = k
    

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y


    def predict(self, X, n_loops=0):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loop(X)
        else:
            distances = self.compute_distances_two_loops(X)
        
        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)


    def compute_distances_two_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        num_test = X.shape[0]
        num_train = self.train_X.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i, j] = np.sum(np.abs(X[i, :] - self.train_X[j, :]))

        return dists


    def compute_distances_one_loop(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        num_test = X.shape[0]
        num_train = self.train_X.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i] = np.sum(np.abs(X[i,:] - self.train_X), axis = 1)

        return dists


    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        return cdist(X, self.train_X, metric='cityblock')


    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        """
        num_test = distances.shape[0]  # I.e. the amount of samples in our test
        prediction = np.zeros(num_test, dtype=object)  # List that for each sample predicts the target
        for n in range(num_test):
            k_neighbours_y = []  # The list with the target values of k nearest neighbours
            neighbours_x = distances[n].argsort()  # We sort the arguments (train samples) by their distance to the test
            for i in range(self.k):
                k_neighbours_y.append(self.train_y[neighbours_x[i]])  # Add only k nearest neighbours' target
            prediction[n] = mode(k_neighbours_y)  # Anf find the mode

        return prediction


    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """

        num_test = distances.shape[0]  # I.e. the amount of samples in our test
        prediction = np.zeros(num_test, dtype=object)  # List that for each sample predicts the target
        for n in range(num_test):
            k_neighbours_y = []  # The list with the target values of k nearest neighbours
            neighbours_x = distances[n].argsort()  # We sort the arguments (train samples) by their distance to the test
            for i in range(self.k):
                k_neighbours_y.append(self.train_y[neighbours_x[i]])  # Add only k nearest neighbours' target
            prediction[n] = mode(k_neighbours_y)  # Anf find the mode

        return prediction
