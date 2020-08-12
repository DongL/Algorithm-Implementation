###########################################
# Kmeans
# Author: Dong Liang
# Aug 12, 2020
###########################################


# 2-D version
class Kmeans(object):
    def __init__(self, k):
        self.k = k
    
    def initialization(self, data):
        self.data = data.reshape(-1, 1)
        self.theta = np.random.choice(self.data.squeeze().astype('float'), self.k, replace = False).reshape(-1, 1)
    
        
    def fit(self, data, epsilon = 0.01):
        # Initialization
        self.initialization(data)
        thetas = [0, 0]
        deltaThetas = 1e6
        
        while np.sum(deltaThetas) > epsilon:
            # E step: calculate the memebership matrix U.
            U = self.predict(self.data)
            
            # M step: update the parameter theta, or the means of clusters.
            self.theta = (U.T @ self.data) / (U.T @ np.ones((self.data.shape[0], 1)))
            
            thetas.append(self.theta)
            deltaThetas = np.abs(thetas[-1] - thetas[-2])
            
    def predict(self, X):
        # Reshape
        X = X.reshape(-1, 1)
        
        # Break ties
        if self.theta[0][0] == self.theta[1][0]:    # breaking ties
            i = np.random.choice(range(self.k), 1)[0] 
            self.theta[i][0] += 1e-6

        # Membership matrix    
        membership = np.argmin((X - self.theta.reshape(1, -1)) ** 2, axis = 1)
        U = np.eye(len(self.data), self.k)[membership]
        return U


# Multidimensional case version
class Kmeans_multi(object):
    def __init__(self, k):
        self.k = k
    
    def initialization(self, data):
        self.data = data 
        # Pick random samples from data matrix to initialize the parameter theta
        indices = np.random.choice(self.data.shape[0], self.k, replace = False) 
        self.theta = self.data[indices, :].astype('float')
    
        
    def fit(self, data, epsilon = 0.01):
        # Initialization
        self.initialization(data)
        thetas = [0, 0]
        deltaThetas = 1e6
        
        while np.sum(deltaThetas) > epsilon:
            # E step: calculate the memebership matrix U.
            U = self.predict(self.data)
            
            # M step: update the parameter theta, or the means of clusters.
            for i in range(self.k):
                self.theta[i,:] = np.sum(U[:, i].reshape(-1, 1) * self.data, axis = 0) / np.sum(U[:, i], axis = 0)
            
            thetas.append(self.theta)
            deltaThetas = np.abs(thetas[-1] - thetas[-2])
            
    def predict(self, X):
        
        # Break ties: 
        for i in range(self.k):
            for j in range(i + 1, self.k):
                if self.theta[i] == self.theta[j] :
                    index = np.random.choice([i, j], 1)[0]
                    self.theta[index] +=1e-6
        
        # Membership matrix
        dist = np.zeros((X.shape[0], self.k))
        for i in range(self.k):
            dist[:, i] = np.sum((X - self.theta[i]) * (X - self.theta[i]), axis = 1)

        membership = np.argmin(dist, axis = 1)
        U = np.eye(X.shape[0], self.k)[membership]
        return U
