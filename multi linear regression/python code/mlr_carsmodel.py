
# Multilinear Regression
import pandas as pd
import numpy as np

# loading the data
a = pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/multi linear regression/Datasets_MLR/ToyotaCorolla.csv",encoding='utf-8')
a=a.drop("Id",axis= True)
a=a.drop('Mfg_Month',axis=True)
a=a.drop('Mfg_Year',axis= True)
a=a.drop('Met_Color',axis= True)
a=a.drop('Automatic',axis= True)
a=a.drop('Cylinders',axis= True)
a=a.drop('Mfr_Guarantee',axis= True)
a=a.drop('BOVAG_Guarantee',axis= True)
a=a.drop('Guarantee_Period',axis= True)
a=a.drop('ABS',axis= True)
a=a.drop('Airbag_1',axis= True)
a=a.drop('Airbag_2',axis= True)
a=a.drop('Airco',axis= True)
a=a.drop('Boardcomputer',axis= True)
a=a.drop('CD_Player',axis= True)
a=a.drop('Central_Lock',axis= True)
a=a.drop('Powered_Windows',axis= True)
a=a.drop('Power_Steering',axis= True)
a=a.drop('Radio',axis= True)
a=a.drop('Mistlamps',axis= True)
a=a.drop('Sport_Model',axis= True)
a=a.drop('Backseat_Divider',axis= True)
a=a.drop('Metallic_Rim',axis= True)
a=a.drop('Radio_cassette',axis= True)
a=a.drop('Tow_Bar',axis= True)
a=a.drop('Fuel_Type',axis= True)
a=a.drop(['Color','Automatic_airco',],axis=True)
a.columns
a.describe()
a.info()
#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# Price
plt.bar(height = a.Price, x = np.arange(1, 1437, 1))
plt.hist(a.Price) #histogram
plt.boxplot(a.Price) #boxplot


# Jointplot
import seaborn as sns
sns.jointplot(x=a['KM'], y=a['Price'])

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(a['KM'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(a.Price, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(a.iloc[:, :])

# Detection of outliers (find limits for price based on IQR)
IQR = a['Price'].quantile(0.75) - a['Price'].quantile(0.25)
lower_limit = a['Price'].quantile(0.25) - (IQR * 1.5)
upper_limit = a['Price'].quantile(0.75) + (IQR * 1.5)

############### 1. Remove (let's trim the dataset) ################
# Trimming Technique
# let's flag the outliers in the data set
outliers_a = np.where(a['Price'] > upper_limit, True, np.where(a['Price'] < lower_limit, True, False))
a_trimmed = a.loc[~(outliers_a), ]
a.shape, a_trimmed.shape

plt.boxplot(a_trimmed.Price) #boxplot for trimmed data

                             
# Correlation matrix 
cor=a_trimmed.corr()



# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('Price ~ Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight' , data = a_trimmed).fit() # regression model

# Summary
ml1.summary()


# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
#index 221 is showing high influence so we can exclude that entire rows
a_new = a_trimmed.drop(a_trimmed.index[[221]])

# Preparing model                  
ml_new = smf.ols('Price ~ Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight', data = a_new).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_Age_08_04 = smf.ols('Age_08_04~KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight', data = a_new).fit().rsquared  
vif_Age_08_04 = 1/(1 - rsq_Age_08_04) 

rsq_KM = smf.ols(' KM~Age_08_04+HP+cc+Doors+Gears+Quarterly_Tax+Weight', data = a_new).fit().rsquared  
vif_KM = 1/(1 - rsq_KM)

rsq_HP = smf.ols('HP~Age_08_04+KM+cc+Doors+Gears+Quarterly_Tax+Weight', data = a_new).fit().rsquared  
vif_HP = 1/(1 - rsq_HP) 

rsq_cc = smf.ols('cc~Age_08_04+KM+HP+Doors+Gears+Quarterly_Tax+Weight ', data = a_new).fit().rsquared  
vif_cc = 1/(1 - rsq_cc) 

rsq_Doors = smf.ols(' Doors~Age_08_04+KM+HP+cc+Gears+Quarterly_Tax+Weight', data = a_new).fit().rsquared  
vif_Doors = 1/(1 - rsq_Doors)

rsq_Gears = smf.ols(' Gears~Age_08_04+KM+HP+cc+Doors+Quarterly_Tax+Weight', data = a_new).fit().rsquared  
vif_Gears = 1/(1 - rsq_Gears)

rsq_Quarterly_Tax = smf.ols(' Quarterly_Tax~Age_08_04+KM+HP+cc+Doors+Gears+Weight', data = a_new).fit().rsquared  
vif_Quarterly_Tax = 1/(1 - rsq_Quarterly_Tax)

rsq_Weight = smf.ols(' Weight~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax', data = a_new).fit().rsquared  
vif_Weight = 1/(1 - rsq_Weight)

# Storing vif values in a data frame
d1 = {'Variables':['Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight'], 'VIF':[vif_Age_08_04, vif_KM,vif_HP,vif_cc,vif_Doors,vif_Gears,vif_Quarterly_Tax,vif_Weight]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As cc is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('Price ~ Age_08_04+KM+HP+Doors+Gears+Quarterly_Tax', data = a_new).fit()
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
sns.residplot(x = pred, y = a_trimmed.Price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
a_train, a_test = train_test_split(a_trimmed, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("Price ~ Age_08_04+KM+HP+Doors+Gears+Quarterly_Tax", data = a_train).fit()

# prediction on test data set 
test_pred = model_train.predict(a_test)

# test residual values 
test_resid = test_pred - a_test.Price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(a_train)

# train residual values 
train_resid  = train_pred - a_train.Price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

