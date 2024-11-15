# import Numpy module
import numpy as np

# Define the K-Means modele classe
class Kmeans:
    """" Modele KMeans creation"""
    def __init__(self,n_clusters):
        self.n_clusters= n_clusters
        self.X = None
        self.centroid = None
        self.labels = None
        self.distances = None
        self.centroid_old = None
        self.min_distances = None


    # centroid initialization function
    def intialize_centroid(self,X):
        """Initialize the centroid from our data X"""
        self.X = X
        self.centroid =  self.X[np.random.choice(self.X.shape[0], self.n_clusters , replace=False), :]
        return self.centroid


    # Distances each points to each centroides computation function
    def computes_distances(self):
        """Compute the distance between the points and the centroids"""
        n = self.X.shape[0]
        k= self.n_clusters
        self.distances = np.zeros((n,k))
        self.centroid = self.intialize_centroid(self.X)
        for i in range(k):
            for j in range(n):
              self.distances[j,i] = np.sum((self.X[j,:]-self.centroid[i,:])**2, axis =0)
        return self.distances
      

    #labels estimations functions 
    def get_labels(self):
        """Generate labels's array """
        self.distances =  self.computes_distances()
        self.labels = np.argmin(self.distances,axis = 1)
        return self.labels
    
    
    #Minimales distances computation function
    def get_mini_distances(self):
        """"return the minimum distances of distances array along each rows"""
        self.distances =  self.computes_distances()
        self.min_distances = np.amin(self.distances,axis = 1)
        return self.min_distances
    


    #Centroids update function
    def update_centroid(self):
        """Updates the centroids positions"""
        self.computes_distances()
        self.centroid_old = self.centroid
        k = self.n_clusters
        n = self.X.shape[1]
        for i in range (0,k): # recompute the centroids
           self.centroid[i,:] = np.mean(self.X[np.where(self.labels == i) ,:],1).reshape(1,n)
        return self.centroid
    
    

    # Fitting function
    def fit(self,X):
        self.X = X
        k = self.n_clusters
        self.centroid = self.intialize_centroid(X)
        self.centroid_old = np.zeros((k,X.shape[1]))
        while np.abs(self.centroid - self.centroid_old).sum() >0.000001:
            self.computes_distances() 
            self.get_labels()
            self.get_mini_distances()
            self.update_centroid()
      
    

    # Predict method
    def predict(self, X_new):
       self.X = X_new
       self.distances=self.computes_distances() 
       self.labels = self.get_labels()
       self.min_distances = self.get_mini_distances()
       self.centroid = self.update_centroid()  
       return self.labels