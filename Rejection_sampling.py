###########################################
# Rejection Sampling
# Author: Dong Liang
# Apr 2, 2021
###########################################

from scipy.stats import norm, uniform
import seaborn as sns 

# Proposal distribution
Q = norm(50, 30)
# Q = uniform(-50, 200)


# Target distribution
P = lambda x: norm.pdf(x, loc=30, scale=10) + norm.pdf(x, loc=80, scale=20)

# Scaling factor
x = np.arange(-50, 151)
k = max(P(x) / Q.pdf(x))

# Initialization
i = 0
N = 5000
sample = list()
U = uniform(0, 1)

# Rejection sampling
for _ in range(5000):
    x = Q.rvs()
    u = U.rvs() # * k * Q.pdf(x)
    
    if u < P(x) / (k * Q.pdf(x)):
        sample.append(x)
    i += 1

 # Plotting 
sns.distplot(sample)
