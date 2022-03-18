
# Q2) a

import pandas as pd

# loading the data
cars = pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/Probability distribution & CLT & confidence interval/probability dataset/probability dataset/Cars.csv")
cars.info()

import scipy.stats as stats
import pylab

# Checking Whether data is normally distributed
stats.probplot(cars.MPG, dist="norm", plot=pylab)



# b)	

import pandas as pd

# loading the data
wc_at = pd.read_csv("C:/Users/me/Downloads/probability dataset (1)/probability dataset/wc-at.csv")
wc_at.info()

import scipy.stats as stats
import pylab

# Checking Whether data is normally distributed
stats.probplot(wc_at.AT, dist="norm", plot=pylab)

stats.probplot(wc_at.Waist, dist="norm", plot=pylab)

##################################################################################
