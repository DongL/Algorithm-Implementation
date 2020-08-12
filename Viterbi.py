
###########################################
# Viterbi algorithm (Hidden Markov Model)
# Author: Dong Liang
# Aug 12, 2020
###########################################

class Viterbi(object):
	'''
	The Viterbi algorithm is implemented within a hidden Markov model framework.
	'''

    def __init__(self, T, E, n_state, P0):
        '''
        - Arguments
            T: transition matrix
            E: emission matrix
            n_state: number of states
            P0: intitial probablilies for each state   
        '''
        self.T = T
        self.E = E
        self.N = n_state
        self.P0 = P0
        self.P_bar = np.empty(self.E.shape)
        self.B = np.empty(self.E.shape)
#         self.hidden_state_sequence = list()

        
    def b(self, s, t):
        return np.argmax(self.T[:, s] * self.P_bar[:, t], axis = 0)
    
    def transition_prob(self, s, t):
        return self.T[self.b(s, t), s]
    
    
    def viterbi(self):
        # Initialization
        self.P_bar[:, 0] = self.P0
#         self.B[:, 0] = [0, 0]
        
        # Viterbi paths
        for s in range(self.N):
            for t in range(self.E.shape[1] - 1):
                self.P_bar[s, t + 1] = self.transition_prob(s, t) * self.P_bar[self.b(s, t), t] * self.E[s, t + 1]
                self.B[s, t + 1] = self.b(s, t) 
                
        # Normalization
        self.P_bar /= self.P_bar.sum(axis = 0, keepdims = True) 
     
    def backtracking(self):
        self.hidden_state_sequence = np.empty(self.E.shape[1])
        
        # check and intialize the last state
        last_state = np.argmax(self.P_bar[:, -1]) 
        self.hidden_state_sequence[-1] = last_state

        for i in range(self.E.shape[1] - 1): 
            state = self.B[int(self.hidden_state_sequence[-1 - i]), -2 - i]
            self.hidden_state_sequence[-2 - i] = int(state)




#################### Self test #####################
# Parameters
T = np.array([[0.9, 0.1], [0, 1]])
E = P_tilde 
n_state = 2
P0 = P_tilde[:, 0]

# Viterbi
vit = Viterbi(T, E, n_state, P0)
vit.viterbi()

# Visualize
plt.imshow(vit.P_bar, aspect = 200, cmap = 'inferno')
plt.xticks(np.arange(1, P_tilde.shape[1], 50))
plt.yticks([0.0, 1.0], ['Piano', 'Clap'])
plt.colorbar()
plt.gcf().set_size_inches(10, 4)
plt.title('Posterior matrix uisng Viterbi algorithm')

# Backtracing
vit.backtracking()
plt.plot(range(len(vit.hidden_state_sequence)), vit.hidden_state_sequence, 'o')
plt.xticks(np.arange(1, P_tilde.shape[1], 50))
plt.yticks([0.0, 1.0], ['Piano', 'Clap'])
plt.gcf().set_size_inches(8, 2)
plt.title('Hidden states by Viterbi backtracking')
