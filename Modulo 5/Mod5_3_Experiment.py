#### LOAD PACKAGES ####
import pandas as pd
import numpy as np
import scipy.stats as ss
import statsmodels.stats.weightstats as ws
from statsmodels.stats.power import tt_ind_solve_power
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

#### LOAD DATA ####
dat = pd.read_csv("datasets/logos.csv")

dat[['friendly', 'inviting', 'interesting', 'positive', 'pleasant']].corr().round(3)
dat['sentiment'] = dat[['friendly', 'inviting', 'interesting', 'positive', 'pleasant']].apply(np.mean, axis = 1)
dat.head()

## Print some summary statistics
print('Mean of Sentiment = ' + str(np.mean(dat.sentiment)))
print('STD of Sentiment = ' + str(np.std(dat.sentiment)))

## Plot a histogram
ax = plt.figure(figsize=(8, 6)).gca() # define axis
dat.sentiment.plot.hist(ax = ax, alpha = 0.6, bins = 15)
plt.title('Histogram of Sentiment')
plt.xlabel('Sentiment')

for col in dat.columns:
    print(col + ' has missing values ' + 
          str((dat[col].isnull().values.any())) or str(dat[col].isna().values.any()))
    
print(dat.shape)
dat.dropna(subset = ['logo'], inplace = True)
print(dat.shape)

## Check once more
print('\n')
for col in dat.columns:
    print(col + ' has missing values ' + 
          str((dat[col].isnull().values.any())) or str(dat[col].isna().values.any()))

ax = plt.figure(figsize=(8,8)).gca() # define axis
sns.boxplot(x = 'logo', y = 'sentiment', data = dat, ax = ax)
sns.swarmplot(x = 'logo', y = 'sentiment', color = 'black', data = dat, ax = ax, alpha = 0.4)
