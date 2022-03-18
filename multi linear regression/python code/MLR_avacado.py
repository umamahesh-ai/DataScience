# Multilinear Regression
import pandas as pd
import numpy as np

# loading the data
a = pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/multi linear regression/Datasets_MLR/Avacado_Price.csv")
a=a.drop(['type','region','year'],axis=True)
a.describe()
a.columns
#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# AveragePrice
plt.bar(height = a.AveragePrice, x = np.arange(1, 18250, 1))
plt.hist(a.AveragePrice) #histogram
plt.boxplot(a.AveragePrice) #boxplot


# Jointplot
import seaborn as sns
sns.jointplot(x=a['Total_Volume'], y=a['AveragePrice'])

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(a['Small_Bags'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(a.AveragePrice, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(a.iloc[:, :])

# Detection of outliers (find limits for price based on IQR)
IQR = a['AveragePrice'].quantile(0.75) - a['AveragePrice'].quantile(0.25)
lower_limit = a['AveragePrice'].quantile(0.25) - (IQR * 1.5)
upper_limit = a['AveragePrice'].quantile(0.75) + (IQR * 1.5)

############### 1. Remove (let's trim the dataset) ################
# Trimming Technique
# let's flag the outliers in the data set
outliers_a = np.where(a['AveragePrice'] > upper_limit, True, np.where(a['AveragePrice'] < lower_limit, True, False))
a_trimmed = a.loc[~(outliers_a), ]
a.shape, a_trimmed.shape

plt.boxplot(a_trimmed.AveragePrice) #boxplot for trimmed data

                             
# Correlation matrix 
cor=a_trimmed.corr()


# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + Xlarge_Bags ' , data = a_trimmed).fit() # regression model

# Summary
ml1.summary()


# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)

#
a_new = a_trimmed.drop(a_trimmed.index[[15560]])
a_new = a_trimmed.drop(a_trimmed.index[[17468]])
# Preparing model                  
ml_new = smf.ols('AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags + Xlarge_Bags ', data = a_new).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_Total_Bags = smf.ols('Total_Bags ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 +  Small_Bags + Large_Bags + Xlarge_Bags', data =a_new ).fit().rsquared  
vif_Total_Bags= 1/(1 - rsq_Total_Bags) 

rsq_Small_Bags = smf.ols('Small_Bags ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Large_Bags + Xlarge_Bags', data = a_new).fit().rsquared  
vif_Small_Bags = 1/(1 - rsq_Small_Bags)

rsq_Large_Bags = smf.ols('Large_Bags ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Xlarge_Bags', data = a_new).fit().rsquared  
vif_Large_Bags = 1/(1 - rsq_Large_Bags) 

rsq_XLargeBags  = smf.ols('Xlarge_Bags ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags', data = a_new).fit().rsquared  
vif_XLargeBags  = 1/(1 - rsq_XLargeBags ) 


# Storing vif values in a data frame
d1 = {'Variables':['Total_Bags','Small_Bags','Large_Bags','Xlarge_Bags'], 'VIF':[vif_Total_Bags, vif_Small_Bags, vif_Large_Bags, vif_XLargeBags]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As Xlarge_Bags is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags', data = a_new).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(a_new)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = a_new.AveragePrice, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
a_train, a_test = train_test_split(a_new, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags", data = a_train).fit()

# prediction on test data set 
test_pred = model_train.predict(a_test)

# test residual values 
test_resid = test_pred - a_test.AveragePrice
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(a_train)

# train residual values 
train_resid  = train_pred - a_train.AveragePrice
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse


