###########################################
# Rejection Sampling
# Author: Dong Liang
# Apr 2, 2021
###########################################

from scipy.stats import norm, uniform
import seaborn as sns 

# Proposal distribution
Q = norm(50, 35)
# Q = uniform(-50, 200)


# Target distribution
P = lambda x: norm.pdf(x, loc=20, scale=15) + norm.pdf(x, loc=90, scale=25)

# Scaling factor
x = np.arange(-150, 150)
k = max(P(x) / Q.pdf(x))

# Initialization
N = 10000
sample = list()
U = uniform(0, 1)

# Rejection sampling
for _ in range(N):
    x = Q.rvs()
    u = U.rvs() # * k * Q.pdf(x)
    if u < P(x) / (k * Q.pdf(x)):
        sample.append(x)

# Plotting 
sns.distplot(sample)
