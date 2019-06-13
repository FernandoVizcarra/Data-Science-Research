#### LOAD PACKAGES ####
from scipy import stats
import scipy.stats as ss
import pandas as pd
import numpy as np
import statsmodels.stats.weightstats as ws
from statsmodels.stats.power import tt_ind_solve_power
import seaborn as sns
import matplotlib.pyplot as plt

#### LOAD DATA ####
import pandas as pd
dat = pd.read_csv("datasets/causal.csv")

# Inspect data
print(dat.columns)

print('\n')
print(dat.head())

print('\n')
print(dat.group.unique())

dat.describe()
dat[['group','prod']].groupby('group').mean()
dat[['group','prod']].groupby('group').std()

def t_test_two_samp(df, alpha, alternative='two-sided'):
    
    a = df[df.group == 'control']['prod']
    b = df[df.group == 'intervention']['prod']    
    
    diff = a.mean() - b.mean()

    res = ss.ttest_ind(a, b)
      
    means = ws.CompareMeans(ws.DescrStatsW(a), ws.DescrStatsW(b))
    confint = means.tconfint_diff(alpha=alpha, alternative=alternative, usevar='unequal') 
    degfree = means.dof_satt()

    index = ['DegFreedom', 'Difference', 'Statistic', 'PValue', 'Low95CI', 'High95CI']
    return pd.Series([degfree, diff, res[0], res[1], confint[0], confint[1]], index = index)   
   

test = t_test_two_samp(dat, 0.05)
test