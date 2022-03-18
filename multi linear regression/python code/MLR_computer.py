# Multilinear Regression
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
# loading the data
a = pd.read_csv("C:\\Users\\me\\Downloads\\Multilinear_ProblemStatement\\Datasets_MLR\\\\Computer_Data.csv")
lb=LabelEncoder()
a['cd']=lb.fit_transform(a['cd'])
a['multi']=lb.fit_transform(a['multi'])
a['premium']=lb.fit_transform(a['premium'])
a=a.drop("Unnamed: 0",axis= True)
a.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# price
plt.bar(height = a.price, x = np.arange(1, 6260, 1))
plt.hist(a.price) #histogram
plt.boxplot(a.price) #boxplot


# Jointplot
import seaborn as sns
sns.jointplot(x=a['speed'], y=a['price'])

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(a['trend'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(a.price, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(a.iloc[:, :])
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

# let's explore outliers in the trimmed dataset
sns.boxplot(a_trimmed.price)
# we see no outiers

                             
# Correlation matrix 
corr=a_trimmed.corr()



# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend', data = a_trimmed).fit() # regression model

# Summary
ml1.summary()

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
#index 4477,3783,5960 is showing high influence so we can exclude that entire rows
a_new = a_trimmed.drop(a_trimmed.index[[4477]])
a_new = a_trimmed.drop(a_trimmed.index[[3783]])
a_new = a_trimmed.drop(a_trimmed.index[[5960]])

# Preparing model                  
ml_new = smf.ols('price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend',data= a_new).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_speed = smf.ols('speed ~ price + hd + ram + screen + cd + multi + premium + ads + trend', data = a_trimmed).fit().rsquared  
vif_speed = 1/(1 - rsq_speed)

rsq_hd = smf.ols(' hd ~ speed + price + ram + screen + cd + multi + premium + ads + trend', data = a_trimmed).fit().rsquared  
vif_hd = 1/(1 - rsq_hd) 

rsq_ram = smf.ols('ram ~ speed + hd + price + screen + cd + multi + premium + ads + trend', data = a_trimmed).fit().rsquared  
vif_ram = 1/(1 - rsq_ram) 

rsq_screen = smf.ols('screen ~ speed + hd + ram + price + cd + multi + premium + ads + trend', data = a_trimmed).fit().rsquared  
vif_screen= 1/(1 - rsq_screen) 

rsq_cd = smf.ols('cd ~ speed + hd + ram + screen + price + multi + premium + ads + trend', data = a_trimmed).fit().rsquared  
vif_cd= 1/(1 - rsq_cd) 

rsq_multi = smf.ols('multi ~ speed + hd + ram + screen + cd + price + premium + ads + trend', data = a_trimmed).fit().rsquared  
vif_multi= 1/(1 - rsq_multi)

rsq_premium = smf.ols('premium ~ speed + hd + ram + screen + cd + multi + price + ads + trend', data = a_trimmed).fit().rsquared  
vif_premium= 1/(1 - rsq_premium) 

rsq_ads= smf.ols('ads ~ speed + hd + ram + screen + cd + multi + premium + price + trend', data = a_trimmed).fit().rsquared  
vif_ads= 1/(1 - rsq_ads) 

rsq_trend = smf.ols('trend ~ speed + hd + ram + screen + cd + multi + premium + ads + price', data = a_trimmed).fit().rsquared  
vif_trend= 1/(1 - rsq_trend) 


 



# Storing vif values in a data frame
d1 = {'Variables':['speed','hd','ram','screen','cd' ,'multi','premium','ads','trend'], 'VIF':[vif_speed,vif_hd,vif_ram,vif_screen,vif_cd ,vif_multi,vif_premium,vif_ads,vif_trend]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As hd is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('price ~ speed + ram + screen + cd + multi + premium + ads + trend', data = a_trimmed).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(a_trimmed)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = a_new.price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
a_train, a_test = train_test_split(a_trimmed, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("price ~ speed + ram + screen + cd + multi + premium + ads + trend", data = a_train).fit()

# prediction on test data set 
test_pred = model_train.predict(a_test)

# test residual values 
test_resid = test_pred - a_test.price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(a_train)

# train residual values 
train_resid  = train_pred - a_train.price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
