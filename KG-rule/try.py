from scipy import stats
import random
import numpy as np

# times = np.random.randint(1,2,1000) + np.random.randint(0,4,1000)
# times = [1 for i in range(10)] + [2 for i in range(10)]
times = np.random.randint(2,4,2000)
times_1 = np.random.randint(2,4,1600)
print(type(times))
print(stats.ks_2samp(times, times_1))
# print(stats.kstest(times, 'uniform'))
# print(stats.uniform(loc=1, scale=1).cdf(6))