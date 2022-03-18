
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
# loading the data
a = pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/Lasso Redige regression/Datasets_LassoRidge/Computer_Data (1).csv")
lb=LabelEncoder()
a['cd']=lb.fit_transform(a['cd'])
a['multi']=lb.fit_transform(a['multi'])
a['premium']=lb.fit_transform(a['premium'])
a=a.drop("Unnamed: 0",axis= True)
a.describe()
# price
plt.bar(height = a.price, x = np.arange(1, 6260, 1))
plt.hist(a.price) #histogram
plt.boxplot(a.price) #boxplot
# Detection of outliers (find limits for price based on IQR)
IQR = a['price'].quantile(0.75) - a['price'].quantile(0.25)
lower_limit = a['price'].quantile(0.25) - (IQR * 1.5)
upper_limit = a['price'].quantile(0.75) + (IQR * 1.5)

############### 1. Remove (let's trim the dataset) ################
# Trimming Technique
# let's flag the outliers in the data set
outliers_a = np.where(a['price'] > upper_limit, True, np.where(a['price'] < lower_limit, True, False))
a_trimmed = a.loc[~(outliers_a), ]
a.shape, a_trimmed.shape

# Correlation matrix 
corr=a_trimmed.corr()


# Sctter plot and histogram between variables
sns.pairplot(a_trimmed) # sp-hp, wt-vol multicolinearity issue
### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
a_train, a_test = train_test_split(a_trimmed, test_size = 0.2) # 20% test data

# Preparing the model on train data 
model_train = smf.ols("price ~ speed + ram + screen + cd + multi + premium + ads + trend", data = a_train).fit()

# Prediction
pred = model_train.predict(a_train)
# Error
resid  = pred - a_train.price
# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse

# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###
from sklearn.linear_model import Lasso
help(Lasso)

lasso = Lasso(alpha = 0.13, normalize = True)

lasso.fit(a_trimmed.iloc[:, 1:], a_trimmed.price)

# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(a.columns[1:]))

lasso.alpha

pred_lasso = lasso.predict(a_trimmed.iloc[:, 1:])

# Adjusted r-square
lasso.score(a_trimmed.iloc[:, 1:], a_trimmed.price)

# RMSE
np.sqrt(np.mean((pred_lasso - a_trimmed.price)**2))


### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
help(Ridge)
rm = Ridge(alpha = 0.4, normalize = True)

rm.fit(a_trimmed.iloc[:, 1:], a_trimmed.price)

# Coefficients values for all the independent vairbales
rm.coef_
rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(a_trimmed.columns[1:]))

rm.alpha

pred_rm = rm.predict(a_trimmed.iloc[:, 1:])

# Adjusted r-square
rm.score(a_trimmed.iloc[:, 1:], a_trimmed.price)

# RMSE
np.sqrt(np.mean((pred_rm - a_trimmed.price)**2))
