###########################################
# KNN
# Author: Dong Liang
# Aug 12, 2020
###########################################

class KNN(object):
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        
    def get_distance(self, x):
        # Use euclidean distance (modifiable with ord value)
        distance = np.linalg.norm(x - self.x_train, ord = 2, axis = 1)
        return distance
    
    def find_knn(self, X):
        
        distance_matrix = np.array([
            self.get_distance(X[i])
            for i in range(X.shape[0])
        ])
        
        # Get KNN index matrix
        k_neighbors_index_matrix = np.argsort(distance_matrix, axis = 1)[:, :self.n_neighbors]
        
        return k_neighbors_index_matrix 
            
    def get_membership(self, k_neighbors_index_matrix):
        
        membership = np.array([
            self.y_train[k_neighbors_index_matrix[i]]
            for i in range(k_neighbors_index_matrix.shape[0])
        ])
        
        return membership
    
    def fit(self, x_train, y_train):

        self.x_train = x_train
        self.y_train = y_train
        
        return self
    
    def predict(self, X):
        
        k_neighbors_index_matrix = self.find_knn(X)
        membership = self.get_membership(k_neighbors_index_matrix)
        
        prediction = np.array([
            np.argmax(np.bincount(membership[i]))
            for i in range(membership.shape[0])
        ])
        
        return prediction
    
    def accuracy(self, y_pred, y_true):
        return np.mean(y_pred == y_true)