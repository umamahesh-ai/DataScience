
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
# loading the data

a = pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/Lasso Redige regression/Datasets_LassoRidge/ToyotaCorolla (1).csv",encoding='latin1')
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
lb=LabelEncoder()
a['Model']=lb.fit_transform(a['Model'])
a=a.rename(columns = {'Age_08_04': 'Age', 'Quarterly_Tax': 'QuaterlyTax'}, inplace = False)

a.columns
a.describe()
a.info()

# Sctter plot and histogram between variables
sns.pairplot(a) # 
### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
a_train, a_test = train_test_split(a, test_size = 0.2) # 20% test data
a_train.columns
# preparing the model on train data 
model_train = smf.ols("Model ~ Price,Age,KM,HP,cc,Doors,Gears,QuaterlyTax,Weight", data = a_train).fit()

# prediction on test data set 
test_pred = model_train.predict(a_test)

# test residual values 
test_resid = test_pred - a_test.Profit
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(a_train)

# train residual values 
train_resid  =train_pred - a_train.Profit
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###
from sklearn.linear_model import Lasso
help(Lasso)

lasso = Lasso(alpha = 0.13, normalize = True)

lasso.fit(a.iloc[:, 1:], a.Profit)

# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(a.columns[1:]))

lasso.alpha

pred_lasso = lasso.predict(a.iloc[:, 1:])

# Adjusted r-square
lasso.score(a.iloc[:, 1:], a.Profit)

# RMSE
np.sqrt(np.mean((pred_lasso - a.Profit)**2))


### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
help(Ridge)
rm = Ridge(alpha = 0.4, normalize = True)

rm.fit(a.iloc[:, 1:], a.Profit)

# Coefficients values for all the independent vairbales
rm.coef_
rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(a.columns[1:]))

rm.alpha

pred_rm = rm.predict(a.iloc[:, 1:])

# Adjusted r-square
rm.score(a.iloc[:, 1:], a.Profit)

# RMSE
np.sqrt(np.mean((pred_rm - a.Profit)**2))
