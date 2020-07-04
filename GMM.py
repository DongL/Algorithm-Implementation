
###########################################
# GMM
# Author: Dong Liang
# Apr 18, 2020
###########################################

class GMM(object):
    def __init__(self, k):
        self.k = k
        self.labels = None
    
    def initialization(self, data):
        self.data = data.reshape(-1, 1)
#         self.U = np.random.random(size = (self.data.shape[0], self.k))
        self.mu = np.random.choice(self.data.squeeze().astype('float'), self.k, replace = False)
        self.p = np.ones(self.k)/2
        self.var = np.ones(self.k)
        
    def fit(self, data, epsilon = 0.01):
        self.initialization(data)
        mus = [0]
        deltaMu = 1e6
        
        while np.sum(deltaMu) > epsilon:
            # E step: calculate the memebership matrix U.
            U = self.predict(self.data)
            
            # M step: update the parameters the means, priors and vars of clusters.
            self.mu = (U.T @ self.data) / (U.T @ np.ones((self.data.shape[0], 1)))
            self.mu = self.mu.squeeze()
            self.p = U.sum(axis = 0) / self.data.shape[0]
            for j in range(self.k):
                self.var[j] =  ((U[:, j].reshape(1, -1) @ (self.data - self.mu[j]) ** 2 ) / np.sum(U[:, j])).squeeze()  # (U[:, j].T @ np.ones((self.data.shape[0], 1))) 
           
            # Delta mu                                                                 
            mus.append(self.mu.copy())
            deltaMu = np.abs(mus[-1] - mus[-2])
        
        self.labels = np.argmax(self.predict(data), axis = 1)
            
    def predict(self, X):
        # Reshape
        X = X.reshape(-1, 1)
        
        # Break ties
        for i in range(self.k):
            for j in range(i + 1, self.k):
                if self.mu[i] == self.mu[j] and self.var[i] == self.var[j]:
                    index = np.random.choice([i, j], 1) 
                    self.mu[index] +=1e-200

        # Membership matrix U
        U = np.random.random(size = (self.data.shape[0], self.k))                
        for j in range(self.k):
            U[:, j] = (self.p[j] * np.exp(-(X - self.mu[j]) ** 2 / (2 * self.var[j]) ) / (np.sqrt((2 * np.pi)**self.k * self.var[j]))).squeeze()
        
        U = U / np.sum(U, axis = 1, keepdims=True)

        return U            
