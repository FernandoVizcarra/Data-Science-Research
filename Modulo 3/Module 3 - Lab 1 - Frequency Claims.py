# MODULE 3 - LAB 1 - FREQUENCY

#### LOAD PACKAGES ####
## Use inline magic command so plots appear in the data frame
#%matplotlib inline

## Next the packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import skew
import statsmodels.stats.api as sms

#### lOAD DATA ####
dat = pd.read_csv("datasets/cupsdat.csv")

# Inspect data
print(dat.columns)
dat.head()

dat['count'].describe()
dat.shape

len(dat['count'])

## Remove rows with nan without making copy of the data frame
dat.dropna(axis=0,inplace=True)

## Now get the counts into a dataframe sorted by the number
count_frame=dat['count'].value_counts()
count_frame = pd.DataFrame({'number':count_frame.index, 'counts':count_frame}).sort_values(by = 'number')

## Compute the percents for each number
n=len(dat['count'])
count_frame['percents']=[100*x/n for x in count_frame['counts']]

## Print as a nice table
count_frame[['number','percents']]

## Add a cumsum dat
count_frame['cumsums']=count_frame['percents'].cumsum()

## Print as a nice table
count_frame[['number', 'percents', 'cumsums']]

## HISTOGRAM
plt.hist(dat['count'])
plt.show()

plt.hist(dat['count'], bins=8)
plt.hist(dat['count'], bins = 8)
plt.title('Frequency of number of cups of coffee consumed')
plt.xlabel('Cups of coffee per day')
plt.ylabel('Frequency')
plt.show()

## we can see this is a modestly skewed distribution:
skew(dat['count'])

## mean and median
print(np.mean(dat['count']))
print(np.median(dat['count']))

## CI confidence interval
sms.DescrStatsW(list(dat['count'])).tconfint_mean()


