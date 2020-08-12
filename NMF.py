###########################################
# NMF
# Author: Dong Liang
# Aug 12, 2020
###########################################

class NMF(object):
    def __init__(self, n_basis = 30):
        # Parameters
        self.n_basis = n_basis
        self.J = list()   
        
        # Initialize W and H
        self.W = None # np.random.rand(self.X.shape[0], self.n_basis)
        self.H = None # np.random.rand(self.n_basis, self.X.shape[1])
        
    def __initialization(self, X, fix_W = False):
        
        self.X = X
        
        if not fix_W:
            self.W = np.random.rand(self.X.shape[0], self.n_basis)
        self.H = np.random.rand(self.n_basis, self.X.shape[1])    

        
    @property    
    def loss(self):
        return 0.5 * np.sum((self.X - self.W @ self.H)**2)
    
   

    def gradient_descent(self, X, iteration = 1000, fix_W = False):

        # Initialize X, W and H 
        self.__initialization(X, fix_W)
        
        # Gradient descent
        for i in range(iteration):   
            
            if not fix_W:
                self.W = self.W * (self.X @ self.H.T / (self.W @ self.H @ self.H.T))
                
            self.H = self.H * (self.W.T @ self.X / (self.W.T @ self.W @ self.H))
            self.J.append(self.loss)
        
        
    def __call__(self, X, **kwargs):

        # Perform gradient descent
        self.gradient_descent(X, **kwargs)
        
        # Plot per iteration losses
        plt.plot(range(len(nmf.J)), nmf.J)
#         plt.show()
