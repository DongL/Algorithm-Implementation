   
###########################################
# Gibbs_sampling
# Author: Dong Liang
# Apr 18, 2020
###########################################
from collections import Counter
from statistics import mode

class Gibbs_sampling_pixel_by_pixel(object):
    def __init__(self, D, D_label, mus, vars, weights, a):
        self.D = D
        self.D_label = D_label.copy()
        self.likelihood_fun = list()
        self.pixel_class_matrix = list()
        self.mus = mus
        self.vars = vars
        self.weights = weights
        self.a = a

        for mu, var in zip(mus, vars):
            self.likelihood_fun.append(norm(mu, np.sqrt(var)))  # likelihood functions
        
        
    def f(self, C, C_neighbor):
        return np.where(C == C_neighbor, 0, self.a)

    def Gaussian_kernel(self, C, C_neighbor, sigma = 1):
        return np.exp(- self.f(C, C_neighbor)**2 / sigma**2)
    
    
    def get_neighbors(self, i, j):
        
        nbs = {
            (i - 1, j - 1), 
            (i - 1, j),
            (i - 1, j + 1), 
            (i, j - 1), 
            (i, j + 1), 
            (i + 1, j - 1), 
            (i + 1, j), 
            (i + 1, j + 1)
        }
        
        nbs = {
            index 
            for index in nbs 
            if index[0] >=0 and index[1]>=0 and index[0] < self.D_label.shape[0] and index[1] < self.D_label.shape[1]
        }    

        return nbs

    def get_neighbors_matrix(self, C_matrix):
        neighbors_matrix = []
        upper_left = C_matrix[:-1, :-1]
        neighbors_matrix.append(np.pad(upper_left, [(1,0), (1,0)], mode = 'reflect'))

        upper = C_matrix[:-1, :]
        neighbors_matrix.append(np.pad(upper_left, [(1,0), (0,0)], mode = 'reflect'))

        upper_right = C_matrix[:-1, 1:]
        neighbors_matrix.append(np.pad(upper_left, [(1,0), (0,1)], mode = 'reflect'))

        left = C_matrix[:, :-1]
        neighbors_matrix.append(np.pad(upper_left, [(0,0), (1,0)], mode = 'reflect'))
        
        right = C_matrix[:, 1:]
        neighbors_matrix.append(np.pad(upper_left, [(0,0), (0,1)], mode = 'reflect'))

        down_left = C_matrix[1:, :-1]
        neighbors_matrix.append(np.pad(upper_left, [(0,1), (1,0)], mode = 'reflect'))

        down = C_matrix[1:, :]
        neighbors_matrix.append(np.pad(upper_left, [(0,1), (0,0)], mode = 'reflect'))

        down_right = C_matrix[1:, 1:]
        neighbors_matrix.append(np.pad(upper_left, [(0,1), (0,1)], mode = 'reflect'))

        return neighbors_matrix


    def __get_MRF_prior_matrix(self, c, C_matrix):
        '''
        c: Class of current pixel
        C_matrix: Pixel classes from previous iteration
        '''
        log_prior_matrix = np.empty(C_matrix.shape)
        for i in range(self.D_label.shape[0]):
            for j in range(self.D_label.shape[1]):
                index_neighbors = self.get_neighbors(i, j)
                neighbors = [C_matrix[index] for index in index_neighbors]
                log_prior = np.sum([np.log(self.Gaussian_kernel(c, nbs)) for nbs in neighbors])
                log_prior_matrix[i, j] = log_prior

        return log_prior_matrix

    def __sample_pixel_class_matrix(self, posterior):
        '''Sample pixel classes from posterior
        '''
        pixel_class_matrix = np.empty(self.D_label.shape)
        for i in range(self.D_label.shape[0]):
            for j in range(self.D_label.shape[1]):
                pixel_class_matrix[i, j] = np.random.choice(range(len(self.mus)), p = posterior[:, i, j])
        return pixel_class_matrix


    def gibbs_sampling(self, C_matrix):
        C = range(len(self.mus)) # classes of pixels
        
        # Likelihood tensor 
        log_likelihood  = np.log([self.likelihood_fun[c].pdf(C_matrix) * w for c, w in zip(C, self.weights)])
       

        # Prior tensor
        log_prior = np.array([self.__get_MRF_prior_matrix(c, C_matrix) for c in C])
        
        # Posterior tensor
        log_posterior = log_likelihood + log_prior
        posterior = np.exp(log_posterior) + 1e-200 # proportional to posterior
        posterior /= posterior.sum(axis = 0, keepdims = True) 
 
        # Sample a class for each pixel based on the per-pixel posterior distribution
        pixel_class_matrix = self.__sample_pixel_class_matrix(posterior)
        
        # Pick the class with largest log_posterior for pixel class 
        # pixel_classe_matrix = np.argmax(log_posterior, axis = 0)
        return pixel_class_matrix

    def plot(self):
        D_quantized = self.label_matrix.copy()
        for i in range(len(self.mus)):
            D_quantized = np.where(self.label_matrix == i, self.mus[i], D_quantized)  
        
        plt.subplot(121)
        plt.imshow(D_quantized, cmap = 'gray')
        plt.subplot(122)
        plt.imshow(self.label_matrix, cmap = 'gray')
        plt.gcf().set_size_inches(10, 10)   

    def __call__(self, burn_in_steps = 100):
        # Initialization
        C_matrix = self.D
        

        # Image segmentation using Gibbs sampling
        for i in range(burn_in_steps):
            C_matrix = self.gibbs_sampling(C_matrix)
            self.pixel_class_matrix.append(C_matrix)
            if i > 0:
                rate = 1 - np.mean(self.pixel_class_matrix[-1] == self.pixel_class_matrix[-2])
                print(rate)

        # Majority voting using 0.1N final samples
        sampler = np.array(self.pixel_class_matrix[-int(0.2*burn_in_steps)::])
        labels = np.empty(self.D.shape)
        # print(sampler.shape, 'sampler')
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                labels[i, j] =  Counter(sampler[:, i, j]).most_common(1)[0][0]  # get the mode

        self.label_matrix = labels
        return labels # labels