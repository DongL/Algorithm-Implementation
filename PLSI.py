
###########################################
# PLSI
# Author: Dong Liang
# Apr 18, 2020
###########################################


class PLSI(object):
    def __init__(self, K = 50, fix_B = False):
        '''
        K: number of topics
        J: cost
        Z: poterior; tensor(word x document x topic)
        B: word x topic (矩阵形式)
        Theta: topic x document
        '''
        self.fix_B = fix_B
        self.K = K
        self.J = list()
        

    def initialization(self, X):
        self.X = X 
        if not self.fix_B:
            self.B = np.random.rand(self.X.shape[0], self.K)
        self.Theta = np.random.rand(self.K, X.shape[1])

        self.Bs = [100 * np.ones((self.X.shape[0], self.K)), 1000 * np.ones((self.X.shape[0], self.K))]
        self.Thetas = [100 * np.ones((self.K, X.shape[1])), 1000 * np.ones((self.K, X.shape[1]))]

    def E_step(self):
        '''Calculate the posterior Z
        E step
        '''
        self.Z = np.array([self.B[:, [k]] @ self.Theta[[k], :] for k in range(self.K)])
        self.Z /= (self.Z.sum(axis = 0, keepdims=True) + 1e-100)
       
        
    def M_step(self, X):
        '''Update B and Theta
        Merged E-M step
        '''
        if not self.fix_B:
            self.B = self.B * ((self.X / (1e-100 + self.B @ self.Theta)) @ self.Theta.T)
            self.B /= np.ones((self.X.shape[0], self.X.shape[0])) @ self.B

        self.Theta = self.Theta * (self.B.T @ (self.X / (1e-100 + self.B @ self.Theta)))
        self.Theta /= np.ones((self.K, self.K)) @ self.Theta

    def fit(self, X, epsilon = 0.01):
        # Initialization
        self.initialization(X)

        # Merged EM step
        # for i in range(iteration): 
        iteration = 0
        while not (np.allclose(self.Bs[-1], self.Bs[-2], atol = epsilon) and \
            np.allclose(self.Thetas[-1], self.Thetas[-2], atol = epsilon)): 
            self.E_step()
            self.M_step(X)
            self.Bs.append(self.B)
            self.Thetas.append(self.Theta)
            iteration +=1
            # print(np.sum((self.Bs[-1] - self.Bs[-2])))
            if iteration % 100 == 0:
                print(f"Iteration: {iteration}")
                # print(self.Z)

        
        print(f"Iteration: {iteration}")
        # print(self.Z)
            # print((self.Z / self.Z.sum(axis = 0)).shape)
            # self.posterior.append(self.Z)
