
###########################################
# Kernel PCA
# Author: Dong Liang
# Apr 19, 2020
###########################################

class KPCA(object):
    
    def __init__(self):
        self.kernel_matrix = np.array(None)
        
    def RBF(self, x, y, sigma = 1):
        return np.exp(- np.linalg.norm(x - y, ord = 2) / sigma**2 )
    
    def get_kernel_matrix(self, X):
        km = list()

        for j in range(X.shape[1]):
            res = [
                self.RBF(X[:, i], X[:, j]) 
                for i in range(X.shape[1])
            ]
            km.append(res)
        
        self.kernel_matrix = np.array(km)
        return km  
    
    def zero_centering(self):
        self.kernel_matrix -= self.kernel_matrix.mean(axis = 0, keepdims = True)
        self.kernel_matrix -= self.kernel_matrix.mean(axis = 1, keepdims = True)
    
        
    def __call__(self, X):
        # Generate kernel_matrix
        self.get_kernel_matrix(X)
        
        # Center and normalize the kernel_matrix
#         self.zero_centering(self.kernel_matrix)
        
        # Do eigendecomposition on kernel_matrix 
        eigenvalue, eigenvector = np.linalg.eigh(self.kernel_matrix)
        eigenvalue = eigenvalue[::-1]
        eigenvector = eigenvector[:, ::-1]
        
        # Compute the coefficients
        coefficients = eigenvector.T @ self.kernel_matrix
        
        return coefficients
        