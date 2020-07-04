
###########################################
# STFT
# Author: Dong Liang
# Apr 18, 2020
###########################################


class STFT(object):
    def __init__(self, N = 1024, overlap_fraction = .5):
        self.N = N
        self.overlap_fraction = overlap_fraction
        self.hop = int(np.floor((1 - self.overlap_fraction) * self.N)) 
            
        
    def DFT_matrix(self):
            
        n, f = np.meshgrid(np.arange(self.N), np.arange(self.N)) 
        
        W_mat = np.exp( -1J * (2 * np.pi * f * n/ self.N))
        
        return W_mat
    
    def inverse_DFT_matrix(self):
        n, f = np.meshgrid(np.arange(self.N), np.arange(self.N)) 

        W_mat = np.exp(1J * (2 * np.pi * f * n/ self.N)) / self.N

        return W_mat
    
    def _signal_win_fn(self, win_fn):
        switcher = {
            'hann': hann, 
            'blackman': blackman
        }
        
        return switcher.get(win_fn, 'No such windown function!')

    def stft(self, data, signal_window = 'hann'):
        
        # Zero padding at both ends
        mod = len(data) % self.N 
        data = np.pad(data, (0 , mod + self.hop), 'constant', constant_values = (0 + 0J, 0 + 0J))
#         data = np.pad(data, (self.hop , mod + self.hop), 'constant', constant_values = (0 + 0J, 0 + 0J))
        
        # Apply Hann window  
        signal_win_fn = self._signal_win_fn(signal_window)
        X = [
            data[i: i + self.N] * signal_win_fn(self.N)  # len(data[i: i + self.N])
            for i in range(0, len(data), self.hop) 
            if len(data[i: i + self.N]) == self.N
        ]
        
        # Transpose X
        X = np.array(X).transpose()
        
        
        # DFT matrix F
        F = self.DFT_matrix()
        
        # Spectrogram
        K = np.int(np.floor(self.N / 2) + 1)
        spectrogram = (F @ X)[:K, :]
        
        # Magnitude
        magnitude = np.abs(spectrogram)
        
        return spectrogram, magnitude
    
    def istft(self, S_hat):
        # Inverse DFT matrix F_star
        F_star = self.inverse_DFT_matrix()
        
        # Recover the full S_hat (spectrogram)
        S_hat_full = np.vstack((S_hat, S_hat[1:-1, :][::-1, :]))
        S_hat_full = F_star @ S_hat_full

        
        # DeDFT
        S_hat_full_transpose = S_hat_full.transpose() # X_hat_transpose 157 x 1024
        size = int(S_hat_full_transpose.shape[0] * self.N / 2) + self.N
        data = np.zeros(size )
        
        for i, j in zip(range(0, size, int(self.hop)), range(S_hat_full_transpose.shape[0])):
            data[i: i + self.N] +=  S_hat_full_transpose[j].real
            
        return data

