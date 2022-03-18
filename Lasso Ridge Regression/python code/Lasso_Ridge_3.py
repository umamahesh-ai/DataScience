

# Multilinear Regression with Regularization using L1 and L2 norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
# loading the data
a=pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/Lasso Redige regression/Datasets_LassoRidge/Life_expectencey_LR.csv")
	
a.info()
a.describe()

a.isna().sum()
# Create an imputer object that fills 'Nan' values
# Mean and Median imputer are used for numeric data (
# Mode is used for discrete data 
# for Mean, Meadian, Mode imputation we can use Simple Imputer or a.fillna()
from sklearn.impute import SimpleImputer
# Mean Imputer 
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
a["Life_expectancy"] = pd.DataFrame(mean_imputer.fit_transform(a[["Life_expectancy"]]))
a["Life_expectancy"].isna().sum()
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
a["Adult_Mortality"] = pd.DataFrame(mean_imputer.fit_transform(a[["Adult_Mortality"]]))
a["Adult_Mortality"].isna().sum()
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
a["Alcohol"] = pd.DataFrame(mean_imputer.fit_transform(a[["Alcohol"]]))
a["Alcohol"].isna().sum()
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
a["Hepatitis_B"] = pd.DataFrame(mean_imputer.fit_transform(a[["Hepatitis_B"]]))
a["Hepatitis_B"].isna().sum()
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
a["BMI"] = pd.DataFrame(mean_imputer.fit_transform(a[["BMI"]]))
a["BMI"].isna().sum()
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
a["Polio"] = pd.DataFrame(mean_imputer.fit_transform(a[["Polio"]]))
a["Polio"].isna().sum()
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
a["Polio"] = pd.DataFrame(mean_imputer.fit_transform(a[["Polio"]]))
a["Polio"].isna().sum()

mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
a["Total_expenditure"] = pd.DataFrame(mean_imputer.fit_transform(a[["Total_expenditure"]]))
a["Total_expenditure"].isna().sum()
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
a["Diphtheria"] = pd.DataFrame(mean_imputer.fit_transform(a[["Diphtheria"]]))
a["Diphtheria"].isna().sum()
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
a["GDP"] = pd.DataFrame(mean_imputer.fit_transform(a[["GDP"]]))
a["GDP"].isna().sum()
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
a["Population"] = pd.DataFrame(mean_imputer.fit_transform(a[["Population"]]))
a["Population"].isna().sum()
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
a["thinness"] = pd.DataFrame(mean_imputer.fit_transform(a[["thinness"]]))
a["thinness"].isna().sum()
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
a["Income_composition"] = pd.DataFrame(mean_imputer.fit_transform(a[["Income_composition"]]))
a["Income_composition"].isna().sum()
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
a["Income_composition"] = pd.DataFrame(mean_imputer.fit_transform(a[["Income_composition"]]))
a["Income_composition"].isna().sum()
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
a["Schooling"] = pd.DataFrame(mean_imputer.fit_transform(a[["Schooling"]]))
a["Schooling"].isna().sum()
a.isna().sum()
sns.boxplot(a.Country)
lb=LabelEncoder()
a['Country']=lb.fit_transform(a['Country'])
a['Status']=lb.fit_transform(a['Status'])
a=a.drop("Year",axis= True)
a=a.drop("thinness_yr",axis= True)
help(a.astype)
# Now we will convert 'float64' into 'int64' type. 
a.Life_expectancy = a.Life_expectancy.astype('int64')
a.Adult_Mortality = a.Adult_Mortality.astype('int64')
a.Alcohol = a.Alcohol.astype('int64')
a.percentage_expenditure = a.percentage_expenditure.astype('int64')
a.Hepatitis_B = a.Hepatitis_B.astype('int64')
a.BMI = a.BMI.astype('int64')
a.Polio = a.Polio.astype('int64')
a.Total_expenditure = a.Total_expenditure.astype('int64')
a.Diphtheria = a.Diphtheria.astype('int64')
a.HIV_AIDS = a.HIV_AIDS.astype('int64')
a.GDP = a.GDP.astype('int64')
a.Population = a.Population.astype('int64')
a.thinness = a.thinness.astype('int64')
a.Income_composition = a.Income_composition.astype('int64')
a.Schooling = a.Schooling.astype('int64')
a.dtypes

# Correlation matrix 
corr=a.corr()


# Sctter plot and histogram between variables
sns.pairplot(a) # sp-hp, wt-vol multicolinearity issue
### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
a_train, a_test = train_test_split(a, test_size = 0.2) # 20% test data

# Preparing the model on train data 
model_train = smf.ols("Life_expectancy ~ Country + Adult_Mortality + infant_deaths + Alcohol + percentage_expenditure + Hepatitis_B + Measles + BMI + under_five_deaths+ Polio + Total_expenditure+Diphtheria+HIV_AIDS+GDP+Population+thinness+Income_composition+Schooling", data = a_train).fit()

# Prediction
pred = model_train.predict(a_train)
# Error
resid  =pred - a_train.Life_expectancy
# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse

# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###
from sklearn.linear_model import Lasso
help(Lasso)

lasso = Lasso(alpha = 0.13, normalize = True)

lasso.fit(a.iloc[:, 1:], a.Life_expectancy)

# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(a.columns[1:]))

lasso.alpha
pred_lasso = lasso.predict(a.iloc[:, 1:])
# Adjusted r-square
lasso.score(a.iloc[:, 1:], a.Life_expectancy)
# RMSE
np.sqrt(np.mean((pred_lasso - a.Life_expectancy)**2))
### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
help(Ridge)
rm = Ridge(alpha = 0.4, normalize = True)
rm.fit(a.iloc[:, 1:], a.Life_expectancy)
# Coefficients values for all the independent vairbales
rm.coef_
rm.intercept_
plt.bar(height = pd.Series(rm.coef_), x = pd.Series(a.columns[1:]))
rm.alpha
pred_rm = rm.predict(a.iloc[:, 1:])
# Adjusted r-square
rm.score(a.iloc[:, 1:], a.Life_expectancy)
# RMSE
np.sqrt(np.mean((pred_rm - a.Life_expectancy)**2))

