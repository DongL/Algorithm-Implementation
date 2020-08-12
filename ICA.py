###########################################
# ICA
# Author: Dong Liang
# Aug 12, 2020
###########################################


def whitening(X, k = 20):
    # Covariance matrix
    Xc = X - X.mean(axis = 1, keepdims = True)
    cov = Xc @ Xc.T / (X.shape[1] - 1)
    
    # Eigenvector
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort eigenvalues in decreasing order, together with the corresponding eigenvectors
    idx = np.argsort(eigenvalues)[::-1]   
    eigenvalues = eigenvalues[idx][:k]
    eigenvectors = eigenvectors[:, idx][:, 0:k]
    
    # Whitening
    W_whitening = np.diag((eigenvalues + 1e-10) ** (-1/2)) @ eigenvectors.T
    whitened = W_whitening @ Xc
    
    return whitened, W_whitening, eigenvalues


def f(x):
    return x**3

def g(x):
    return np.tanh(x)

def ICA(Z, learning_rate = 1e-9, epsilon = 0.1):
    N = Z.shape[1]
    K = Z.shape[0]
    W = np.random.normal(size = (K, K)) 
    Y = np.random.normal(size = (K, N))
    deltaW = np.random.normal(size = (K, K))
    sumOfAbsDeltaWs = list()
    count = 0
    
    while np.sum(np.abs(deltaW)) > epsilon:
#     for i in range(200):
        count += 1
        if count % 500 == 0:
            print(f'Iteration: {count}', f'The sum of absolute ∆W: {np.sum(np.abs(deltaW))}')
        
        # Weight
        deltaW = (N * np.identity(K) - g(Y) @ f(Y).T) @ W
        sumOfAbsDeltaWs.append(np.sum(np.abs(deltaW)))
        
        # Update
        W += learning_rate * deltaW
        Y = W @ Z 
    
    print(f'Iteration: {count}', f'The sum of absolute ∆W: {np.sum(np.abs(deltaW))}')    
    
    return Y, sumOfAbsDeltaWs    