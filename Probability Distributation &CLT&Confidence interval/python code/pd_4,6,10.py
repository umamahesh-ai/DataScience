
## Q4)

n = 25
df = n -1 


import scipy.stats as stats

# 95% confidence interval, 
confidence1 = 0.95
t_95 = stats.t.ppf((1 - confidence1)/2 , df)
t_95  # -2.0638985616280205

stats.t.ppf(0.025, 24) 

# 96% confidence interval
confidence2 = 0.96
t_96 = stats.t.ppf((1 - confidence2)/2 , df)
t_96  # -2.1715446760080677


# 99% confidence interval
confidence3 = 0.99
t_99 = stats.t.ppf((1 - confidence3)/2 , df)
t_99  # -2.796939504772804

stats.t.ppf(0.005, 24) 
stats.t.ppf((1 - confidence3)/2 , 24) 

(1-0.99)/2
# t-distribution
# stats.t.cdf(1.98, 24) # Given a value, find the probability; # similar to pt in R
# stats.t.ppf(0.025, 24) # Given probability, find the t value; # similar to qt in R

##### 6

import scipy.stats as stats
p2 = stats.norm.cdf(50, 45, 8)
1-p2

########## 10
import numpy as np
from scipy import stats
from scipy.stats import norm

# Profit1 ~ N(5, 3^2) and 
# Profit2 ~ N(7, 4^2)

Mean = 5+7
Mean
print('Mean Profit is Rs', Mean*45,'Million') # 540

SD = np.sqrt((9)+(16))
SD
print('Standard Deviation is Rs', SD*45, 'Million') # 225

print('Range is Rs',(stats.norm.interval(0.95,540,225)),'in Millions')


X1 = 540+(-1.645)*(225)
X1
print('5th percentile of profit (in Million Rupees) is',np.round(X1,))


stats.norm.cdf(0,5,3)

stats.norm.cdf(0,7,4)




