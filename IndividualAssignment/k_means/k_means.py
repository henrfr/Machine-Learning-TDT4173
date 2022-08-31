import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:
    
    def __init__(self, k: int = 2, init: str = "random", max_iter: int =20, alpha: float =0.2, best_of: int = 5):
        """Sets parameters used for clustering, optimization and animation.

        Args:
            k (int, optional): The number of clusters. Defaults to 2.
            init (str, optional): The centroid initialization method used. Defaults to "random".
            max_iter (int, optional): The number of iterations the algorithm will run. Defaults to 20.
            alpha (float, optional): A scaling parameter for random motion of centroids. Defaults to 0.2.
            best_of (int, optional): The fit() method will do best_of number of iterations and choose the best centroids. Defaults to 5.
        """
        self.k = k
        self.centroids = "Centroids are not set"
        self.init_centroids = init
        self.max_iter = max_iter
        self.centroid_history = ""
        self.alpha= alpha
        self.best_of = best_of
        self.best_centroids = ""
        self.best_distortion = np.inf
        self.best_centroid_history = ""
        
    def fit(self, X: pd.DataFrame):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        X = X.to_numpy()
        self.centroids = np.zeros(shape = (self.k, X.shape[1]))
        self.centroid_history = np.zeros(shape=(self.max_iter+1, self.centroids.shape[0], self.centroids.shape[1]))
        self.best_centroids =np.zeros_like(self.centroids)
        for l in range(self.best_of):

            # Randomly initialize the centroids
            if self.init_centroids == "random":
                for i in range(X.shape[1]):
                    self.centroids[:,i] = np.random.uniform(low=np.min(X[:,i]), high=np.max(X[:,i]), size=self.k)
            # Do a k-means++ like initialization of centroids.
            else:
                visited_idx = []
                init_rnd = np.random.randint(0, high=X.shape[0])
                visited_idx.append(init_rnd)
                self.centroids[0] = X[init_rnd]
                for i in range(1, self.k):
                    X_ = np.delete(X, visited_idx, axis=0)
                    distances = cross_euclidean_distance(self.centroids[:i], X_)
                    idx = np.unravel_index(np.argmax(distances, axis=None), distances.shape)[1]
                    visited_idx.append(idx)
                    self.centroids[i] = X[idx]

            # Makes a history for animations
            self.centroid_history[0] = self.centroids

            # Optimizes the centroid positions
            for _ in range(self.max_iter):
                # Finds the distance from every point to every centroid
                distances = cross_euclidean_distance(self.centroids, X)
                # Finds the indexes of the minimum distances. The int at an index in minimums corresponds to the centroid the index/point belongs to. 
                minimums = np.argmin(distances, axis=0)

                # For all centroid points
                for i in range(self.k):
                    # get the indexes of the points closest to the centroid
                    indices = np.where(minimums == i)[0]

                    # Place all closest points in an array like X
                    centroid_points = np.zeros(shape=(len(indices), X.shape[1]))
                    for j, index in enumerate(indices):
                        centroid_points[j] = X[index]
                    # If the centroid have any points that are closest, necessary to avoid divideByZero.    
                    if len(centroid_points):
                        # The new centroid position is the mean of the points closest to it.
                        self.centroids[i] = np.mean(centroid_points, axis=0)
                # Represents the random motion to avoid local minima, should be improved!
                self.centroids = self.centroids + self.alpha*(np.exp(-_))*np.reshape(np.random.uniform(low=-np.max(X), high=np.max(X), size=self.centroids.shape[0]*self.centroids.shape[1]), self.centroids.shape)
                self.centroid_history[_+1] = self.centroids

            # Calculates the distortion of an optimized initialization   
            distances = cross_euclidean_distance(self.centroids, X)
            prediction = np.argmin(distances, axis=0)
            temp_distortion = euclidean_distortion(X, prediction)
            if temp_distortion < self.best_distortion:
                self.best_centroids = np.copy(self.centroids)
                self.best_distortion = temp_distortion
                self.best_centroid_history = np.copy(self.centroid_history)
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        X = X.to_numpy()
        distances = cross_euclidean_distance(self.best_centroids, X)
        prediction = np.argmin(distances, axis=0)
        return prediction
    
    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return self.best_centroids
    
    def get_centroid_history(self):
        return self.best_centroid_history
    
    
# --- Some utility functions 


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    for c in np.unique(z):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()
        
    return distortion

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    Compute Euclidean distance between two sets of points 
    
    Args:
        x (array<m,d>): float tensor with pairs of 
            n-dimensional points. 
        y (array<n,d>): float tensor with pairs of 
            n-dimensional points. Uses y=x if y is not given.
            
    Returns:
        A float array of shape <m,n> with the euclidean distances
        from all the points in x to all the points in y
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))
