
###########################################
# Multidimensional Scaling (MDS)
# Author: Dong Liang
# Aug 12, 2020
###########################################

class MDS(object):
    def __init__(self):
        self.distance_matrix = np.array(None)
   
    def zero_centering(self):
        # Subtract averages of rows and columns to get to the effect of x.t @ x   
        self.distance_matrix -= self.distance_matrix.mean(axis = 0, keepdims = True)
        self.distance_matrix -= self.distance_matrix.mean(axis = 1, keepdims = True)
        self.distance_matrix = self.distance_matrix * (- 1 / 2)
    
    def __call__(self, distance_matrix):
        # Get distance matrix
        self.distance_matrix = distance_matrix

        # Center the distance/kernel_matrix
        self.zero_centering()
        
        # Do eigendecomposition on the centered distance/kernel_matrix 
        eigenvalue, eigenvector = np.linalg.eigh(self.distance_matrix)
        eigenvalue = eigenvalue[::-1]
        eigenvector = eigenvector[:, ::-1]
        
        # Plot the eigenvalue
        plt.bar(range(len(eigenvalue[:10])), eigenvalue[:10])
        
        # Compute the coefficients
        coefficients = eigenvector @ np.sqrt(np.diag(eigenvalue))
        
        return coefficients