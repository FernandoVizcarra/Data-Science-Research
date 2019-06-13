# MODULO 2 LAB 2 P-VALUES

# THE NULL HYPOTHESIS

# set seed to make random number generation reproducible
import numpy as np
import numpy.random as nr
from scipy import stats
nr.seed(51120122)

#collect a sample of 100 males
males = nr.normal(5, 3, 100)

#collect a sample of 100 females
females = nr.normal(5, 3, 100)

print(np.mean(males))
print(np.mean(females))

difference = males - females
print(difference)

nr.seed(4455)
attitude = nr.normal(2.4, 2.0, 100)

def t_one_sample(samp,mu=0.0, alpha=0.05):
    ''' Function for two-sided one-sample t test'''
    t_stat=stats.ttest_1samp(samp,mu)
    scale=np.std(samp)
    loc=np.mean(samp)
    ci=stats.t.cdf(alpha/2,len(samp),loc=mu,scale=scale)
    print('Results of one-sample two-sided t test')
    print('Mean         = %4.3f' % loc)
    print('t-Statistic  = %4.3f' % t_stat[0])
    print('p-value      < %4.3e' % t_stat[1])
    print('On degrees of freedom = %4d' % (len(samp) - 1))
    print('Confidence Intervals for alpha =' + str(alpha))
    print('Lower =  %4.3f Upper = %4.3f' % (loc - ci, loc + ci))
t_one_sample(difference)