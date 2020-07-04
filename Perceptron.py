
###########################################
# Perceptron
# Author: Dong Liang
# Apr 19, 2020
###########################################

class Activation2(object):
    def __init__(self, activation): 
        if activation == 'sigmoid':
            self.activation = self.sigmoid
            self.dActivation = self.dSigmoid
        
        elif activation == 'softmax':
            self.activation = self.softmax
            self.dActivation = self.dSoftmax
        
        elif activation == 'TanH':
            self.activation = self.TanH
            self.dActivation = self.dTanH
        
        elif activation == 'ReLU':
            self.activation = self.ReLU
            self.dActivation = self.dReLU
    
    # Sigmoid
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def dSigmoid(self, f):
        return f * (1 - f)
    
    # Softmax
    def softmax(self, z):
        return np.exp(z) / np.exp(z).sum(axis = 0)

    def dSoftmax(self, f):
        return f - f**2

    # TanH
    def TanH(self, z):
        # return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        return (2 / (1 + np.exp(-z))) - 1

    def dTanH(self, f):
        return 1 - f**2
    
    # ReLU
    def ReLU(self, z):
        if z <= 0:
            return 0
        else:
            return z
        
    def dReLU(self, z):
        if z <= 0:
            return 0
        else:
            return 1


class Loss(object):
    def __init__(self, loss): 
        if loss == 'cross_entropy':
            self.loss = self.cross_entropy
            self.dLoss = self.dCross_entropy
        elif loss == 'MSE':
            self.loss = self.MSE
            self.dLoss = self.dMSE
        elif loss == 'weighted_error':
            self.loss = self.weighted_error
            self.dLoss = self.dWeighted_error

    def cross_entropy(self, y, y_hat, w = None):
        # print(y, y_hat)
        # loss = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        loss = - np.sum(y * np.log(y_hat), axis = 0) 
        loss = np.mean(loss)
        return loss

    def dCross_entropy(self, y, y_hat, w = None):
        dLoss = (y_hat - y) / (y_hat - y_hat * y_hat)
        return dLoss

    def MSE(self, y, y_hat, w = None):
        loss = np.mean(0.5 * (y_hat - y)**2)
        return loss

    def dMSE(self, y, y_hat, w = None):
        dLoss = y_hat - y
        return dLoss

    def weighted_error(self, y, y_hat, w):
        loss = np.sum(w * (y_hat - y)**2)
        return loss

    def dWeighted_error(self, y, y_hat, w):
        dLoss = w * (y_hat - y)
        return dLoss

class Perceptron2(object):
    
    def __init__(self, activation_fn, loss_fn):
        self.losses = list()
        self.accs = list()
        self.activation_fn = activation_fn
        self.activation = Activation(self.activation_fn).activation 
        self.dActivation = Activation(self.activation_fn).dActivation
        self.loss_fn = loss_fn
        self.loss = Loss(self.loss_fn).loss
        self.dLoss = Loss(self.loss_fn).dLoss
        
    def update_w(self, X, y, w):
        # y_hat = self.softmax(self.w.T @ X)
        
        # Prediction
        y_hat = self.activation(self.w.T @ X)

        # Derivative of activation
        dActivation = self.dActivation(y_hat)
        
        # Derivative of MSE loss
        dLoss = self.dLoss(y, y_hat, w)
        gradient_w =  X @ (dLoss * dActivation).T  #* self.y_hat * ((1 - self.activation(self.w.T @ X)))).T
        self.w = self.w - self.learning_rate * gradient_w
   
    def generate_mini_batches(self, batch_size = 30):
        mini_batches = [
            (self.X_augmented[:, i:i + batch_size], self.y[:, i:i + batch_size], self.sample_weights[:, i:i + batch_size]) 
            for i in range(0, self.y.shape[1], batch_size)
            # if self.y[:, i:i + batch_size].shape[1] == batch_size
        ]

        return np.array(mini_batches)


    def normalization(self, X):
        mean = np.mean(X, axis = 1, keepdims = True)
        sd = np.std(X, axis = 1, keepdims = True)
        return (X - mean) / sd

    def data_preprocessing(self, X):
        # Normalization
        X = self.normalization(X)
        
        # Augmentation
        X = np.vstack((X, np.ones((1, X.shape[1]))))
        
        return X

    def fit(self, X, y_true, batch_size = 30, epoch = 100, learning_rate = 0.01, sample_weights = None, verbose = False):
        '''
        X: Dimension x N
        y: Dimension x N
        '''
        # Initialization
        self.w = np.random.normal(loc=0.0, scale=0.05, size = (X.shape[0] + 1, y_true.shape[0]))
        self.batch_size = 30
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.X = X
        self.y = y_true

        if isinstance(sample_weights, (list, np.ndarray)):
            self.sample_weights = np.array(sample_weights).reshape(1, -1)
        else:
            N = X.shape[1]
            self.sample_weights = np.array([1/N] * N).reshape(1, -1)  
            

        # Data preprocessing
        self.X_augmented = self.data_preprocessing(self.X)


        # Training
        for ep in range(int(self.epoch + self.epoch/10)):
            # Data shuffling
            random_index = np.random.choice(range(y_true.shape[1]), size = y_true.shape[1], replace = False)
            self.X_augmented = self.X_augmented[:, random_index]
            self.y = self.y[:, random_index]
            self.sample_weights = self.sample_weights[:, random_index]
            
            # Data augmentation
            # self.X_augmented = np.vstack((self.X, np.ones((1, self.X.shape[1])))) 
            
            # Generate mini-batches
            mini_batches = self.generate_mini_batches(batch_size)   

            for _, (x, y, w) in enumerate(mini_batches):
                # Update weight
                self.update_w(x, y, w)
            
            # Compute loss
            # y_hat = self.softmax(self.w.T @ self.X_augmented)
            y_hat = self.activation(self.w.T @ self.X_augmented)

            loss = self.loss(self.y, y_hat, self.sample_weights)
            self.losses.append(loss)

            # Compute accuracy
            acc = self.generate_accuracy(self.y, y_hat)
            self.accs.append(acc)

            # Print output
            if verbose:
                if ep % int(self.epoch/10) == 0 or ep == self.epoch - 1: 
                    print(f"Epoch: {ep}")
                    print(f"Loss: {self.losses[ep]}, ", f"Acc: {self.accs[ep]}")

    def predict(self, X):
        # Data augmentation
        # X_augmented = np.vstack((X, np.ones((1, X.shape[1]))))
        X_augmented = self.data_preprocessing(X)
        # print(self.w.T @ X_augmented)
        y_hat = self.activation(self.w.T @ X_augmented)
        return y_hat
    
    def generate_accuracy(self, y, y_hat):
        if y.shape[0] > 1:
            y_hat = np.argmax(y_hat, axis = 0)
            y = np.argmax(y, axis = 0)
            return np.mean(y == y_hat)

        if self.activation_fn == 'TanH':
            y_hat = np.where(y_hat > 0, 1, -1)
        elif self.activation_fn == 'sigmoid':
            y_hat = np.where(y_hat > 0.5, 1, 0)

        return np.mean(y == y_hat)
      