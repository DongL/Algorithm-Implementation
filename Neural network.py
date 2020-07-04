
class Activation(object):
    def __init__(self): 
        pass
        
    def activation(self, activation):
        if activation == 'sigmoid':
            return self.sigmoid
        
        elif activation == 'TanH':
            return self.TanH
        
        elif activation == 'ReLU':
            return self.ReLU
    
    def dActivation(self, activation):
        if activation == 'sigmoid':
            return self.dSigmoid
        
        elif activation == 'TanH':
            return self.dTanH
        
        elif activation == 'ReLU':
            return self.dReLU
    
    # Sigmoid
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z) + 1e-100)

    def dSigmoid(self, f):
        return f * (1 - f)
    
    # TanH
    def TanH(self, z):
        # return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z) + 1e-100)
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

    def cross_entropy(self, y, y_hat):
        # print(y, y_hat)
        # loss = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        loss = - np.sum(y * np.log(y_hat), axis = 0) 
        loss = np.mean(loss)
        return loss

    def dCross_entropy(self, y, y_hat):
        dLoss = (y_hat - y) / (y_hat - y_hat * y_hat)
        return dLoss

    def MSE(self, y, y_hat):
        loss = np.mean(0.5 * (y_hat - y)**2)
        return loss

    def dMSE(self, y, y_hat):
        dLoss = y_hat - y
        return dLoss




class NeuralNetwork2(object):
    def __init__(self, learning_rate = 0.01, n_nodes = 50, output_dim = 1):
        self.learning_rate = learning_rate
        self.n_nodes = n_nodes
        self.output_dim = output_dim
        self.losses = list()
        self.accs = list()
        

    
    def forward(self, X = None, y = None): 
        '''Forward'''
        if isinstance(X, np.ndarray):
            x1 = X
        else:
            x1 = self.X

        z1 = self.W1 @ x1  
        a2 = self.activation(self.activation_fn)(z1)
        z2 = self.W2 @ a2  
        y_hat = self.activation('sigmoid')(z2)

        return y_hat, a2, z2, x1, z1, y
    
    def backward(self, ep, params):  
        
        '''Backpropagation'''

        y_hat, a2, z2, x1, z1, y = params
        
        dLoss = self.dLoss(y, y_hat)
        BP_error2 = dLoss * self.dActivation('sigmoid')(y_hat)
        dW2 = BP_error2 @ a2.T 

        # Gradient W1
        BP_error1 = (self.W2.T @ BP_error2) * self.dActivation(self.activation_fn)(a2) 
        dW1 = BP_error1 @ x1.T

        # Update W1, W2, b1, b2
        self.W2 -= self.learning_rate * dW2
        self.W1 -= self.learning_rate * dW1
        
    def generate_loss(self, y, y_hat, loss = 'cross_entropy'):
        if loss == 'cross_entropy':
            # loss = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) 
            loss = - np.sum(y * np.log(y_hat), axis = 0) 

            # print(y, 'y', y_hat)
            loss = np.mean(loss)
            
        elif loss == 'mse':
            loss = np.mean(0.5 * (y_hat - y)**2)
        
        return loss
    
    def normalization(self, X):
        mean = np.mean(X, axis = 1, keepdims = True)
        sd = np.std(X, axis = 1, keepdims = True)
        return (X - mean) / sd
        
    
    def generate_mini_batches(self, batch_size = 30):

        mini_batches = [
            (self.X_augmented[:, i:i + batch_size], self.y[:, i:i + batch_size]) 
            for i in range(0, self.y.shape[1], batch_size)
            if self.y[:, i:i + batch_size].shape[1] == batch_size
        ]

        return np.array(mini_batches)
            
    def data_preprocessing(self, X):
        # Normalization
        X = self.normalization(X)
        
        # Augmentation
        X = np.vstack((X, np.ones((1, X.shape[1]))))
        
        return X
    
    def fit(self, X, y_true, epoch, activation_fn, loss_fn, valid_X, valid_y, batch_size):
       
        # Initialization
        self.W1 = np.random.normal(loc=0, scale=0.1, size = (self.n_nodes, X.shape[0] + 1)) # / np.sqrt(2/X.shape[0])
        self.W2 = np.random.normal(loc=0, scale=0.1, size = (self.output_dim, self.n_nodes)) # / np.sqrt(2/self.n_nodes)
        self.X = X
        self.y = y_true
        self.valid_X = valid_X
        self.valid_y = valid_y
        self.epoch = epoch

        self.activation_fn = activation_fn
        self.activation = Activation().activation
        self.dActivation = Activation().dActivation
        self.loss_fn = loss_fn
        self.loss = Loss(self.loss_fn).loss
        self.dLoss = Loss(self.loss_fn).dLoss

        
        # Data preprocessing
        self.X_augmented = self.data_preprocessing(self.X)

        # Epoch
        for ep in range(int(self.epoch + self.epoch/10)):
            
            # Data shuffling
            random_index = np.random.choice(range(y_true.shape[1]), size = y_true.shape[1], replace = False)
            self.X_augmented = self.X_augmented[:, random_index]
            self.y = self.y[:, random_index] 
  
            # Make mini-batches
            mini_batches = self.generate_mini_batches(batch_size)
            
            for _, (x, y) in enumerate(mini_batches):
                # Forward/backward propagation
                params = self.forward(x, y)
                y_hat = params[0]

                # update weights
                self.backward(ep, params)
            
            # Epoch losses / accuracies
            y_hat, *_  = self.forward(self.X_augmented)
            loss = self.loss(self.y, y_hat)
            self.losses.append(loss)

            acc = self.generate_accuracy(self.y, y_hat)
            self.accs.append(acc)
            
            # Print outputs
            if ep % int(self.epoch/10) == 0 or ep == self.epoch - 1: 
                print('Epoch', ep)
                print(f"Loss: {self.losses[ep]}, ", f"Acc_train: {self.accs[ep]}, ", f"Acc_valid: {self.accuracy('valid')}")
        

    def predict(self, X):
        X_augmented = self.data_preprocessing(X)
        y_hat, *_  = self.forward(X_augmented)
        return y_hat
#         return np.where(y_hat >= 0.5, 1, 0) 
    
    def generate_accuracy(self, y, y_hat):
        return np.mean(np.where(y_hat >=0.5, 1, 0) == y)
    
    def accuracy(self, type = 'train'):
        if type == 'train':
            return np.mean(np.where(self.predict(self.X) >= 0.5, 1, 0) == self.y)
        elif type == 'valid':
            return np.mean(np.where(self.predict(self.valid_X)>= 0.5, 1, 0) == self.valid_y)