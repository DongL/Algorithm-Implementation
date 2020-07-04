
###########################################
# AdaBoost
# Author: Dong Liang
# Apr 21, 2020
###########################################

import bisect
class AdaBoosts(object):
    def __init__(self, model):
        self.model = model
        self.weak_learners = list()
        self.ws = list()
        self.betas = list()
        self.basis_function = lambda X: 0
        


    def predict(self, X):

        pred = 0
        for beta, wl in zip(self.betas, self.weak_learners):
            pred += beta * wl.predict(X) 

        return pred


    def exponential_loss(self):
        y_hat = np.sign(self.predict(self.X))
        loss = np.sum(np.exp(- self.y * y_hat), axis = 0)
        loss = np.mean(loss)
        return loss


    def weighted_sampling(self, sample_weight):
        # Normalized sample weights
        sample_weight_normalized = sample_weight / np.sum(sample_weight)
        CDF = np.cumsum(sample_weight_normalized)
        N = self.y.shape[1]

        index = np.array([bisect.bisect(CDF, np.random.random(1)) for _ in range(N)])
        X_weighted = self.X[:, index]
        y_weighted = self.y[:, index]
        return X_weighted, y_weighted
            

    def train_weak_learner(self, sample_weights):
        # X, y = self.weighted_sampling(sample_weights)
        
        params = {
            'X': self.X, 
            'y_true': self.y, 
            'epoch': 5, 
            'batch_size': 64,
            'learning_rate':  1500, # 10
            'sample_weights': sample_weights,
            'verbose': False
        }

        weak_learner = self.model(loss_fn = 'weighted_error', activation_fn = 'TanH')
        weak_learner.fit(**params)

        return weak_learner
        
    

    def fit(self, X, y, M = 100):
        N = X.shape[1]

        # Initialization
        self.X = X
        self.y = y
        
        # Initialize sample weights
        self.ws.append([1/N] * N)

        # Initialize the 1st weak learner
        weak_learner = self.train_weak_learner(sample_weights = self.ws[-1])
        self.weak_learners.append(weak_learner)

        # Forward stagewise additive modeling
        for m in range(M):
            # Weak learner prediction
            y_hat = self.weak_learners[-1].predict(self.X)

            # Mixing weight for the weak learner - beta
            misclassification = np.average(self.y != np.sign(y_hat), axis = 1, weights = self.ws[m])
            beta_m = 0.5 * np.log((1 - misclassification) / misclassification)
            self.betas.append(beta_m)
            
            # Sample weights for (m + 1)th weak learner - w
            w = self.ws[m] * np.exp(-beta_m * self.y * y_hat)  # w_(m+1)
            self.ws.append(w)

            # Train the m + 1 th weak learner
            weak_learner = self.train_weak_learner(sample_weights = self.ws[-1])
            self.weak_learners.append(weak_learner)

            # AdaBoost model objective error
            y_hat_ada = self.predict(X)
            acc = np.mean(np.sign(y_hat_ada) == self.y)

            loss = self.exponential_loss()
            # print(acc, 'acc')
            if m % int(M/10) == 0 or m == M - 1: 
                print(f"Iteration: {m}")
                print('Loss', loss, 'Accuracy', acc)
           

def plot_adaBoost(X, y_label, model, step, spacer):

    x1 = np.linspace(X[0, :].min() - spacer, X[0, :].max() + spacer, step)
    x2 = np.linspace(X[1, :].min() - spacer, X[1, :].max() + spacer, step)
    xx, yy = np.meshgrid(x1, x2)
    data = np.array([xx.ravel(), yy.ravel()])
    z = model.predict(data).reshape(xx.shape)

    colors = y_label.flatten()
    weights = model.ws[-1] * 5e4
    plt.pcolormesh(xx, yy, z, cmap = plt.cm.viridis)
    plt.scatter(X[0], X[1], c = colors, cmap = plt.cm.viridis, s = weights, alpha = 0.7)
    plt.gcf().set_size_inches(8,8)
    