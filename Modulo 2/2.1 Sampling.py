import matplotlib.pyplot as plt
import numpy.random as nr
import numpy as np
from scipy import stats

# Generate a random sample data
nr.seed(12345)
# normal(mean, std, n)
x_normal=nr.normal(10,2,50)
#plt.hist(x_normal, bins=10)
#plt.show
#print(x_normal.size)

# Generate a random sample data
nr.seed(3344)
# binomial(n,prob,size=1)
x_binomial=nr.binomial(1,0.3,50)

print(np.sum(x_binomial)/x_binomial.size)
print(stats.itemfreq(x_binomial))

result=[np.mean(nr.binomial(1,0.3,700)) for _ in range(1000)]

x_binomial_mean=np.mean(result)
print()
plt.hist(result)
plt.vlines(0.3, 0.0, 28.0, color = 'red')
plt.vlines(x_binomial_mean, 0.0, 28.0, color = 'black')
plt.xlabel('Results') 
plt.ylabel('Frequency')
plt.title('Histogram of results')

result_errors=[round(x-0.3,2) for x in result]
#print(result_errors)

std_error=np.std(result_errors)
print(std_error)