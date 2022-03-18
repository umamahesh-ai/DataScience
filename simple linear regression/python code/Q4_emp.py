import pandas as pd
import numpy as np
emp=pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/Simple linear regresion/Datasets_SLR/emp_data.csv")
emp.columns
# Exploratory data analysis:
# 1. Measures of central tendency
####calculate the mean 
emp.Salary_hike.mean()
emp.Churn_out_rate.mean()
####calculate the median
emp.Salary_hike.median()
emp.Churn_out_rate.median()
# 2. Measures of dispersion
#calculate the std
emp.Salary_hike.std()
emp.Churn_out_rate.std()
#calculate variance
emp.Salary_hike.var()
emp.Churn_out_rate.var()
#calculate the Range
range_output=max(emp.Salary_hike)-min(emp.Salary_hike)
range_input=max(emp.Churn_out_rate)-min(emp.Churn_out_rate)
# 3. Third moment business decision
emp.Salary_hike.skew()
emp.Churn_out_rate.skew()
# 4. Fourth moment business decision
emp.Salary_hike.kurt()
emp.Churn_out_rate.kurt()
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, 
emp.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 
#Graphical Representation of a output(y=Salary_hike)
plt.bar(height = emp.Salary_hike, x = np.arange(len(emp)))
plt.hist(emp.Salary_hike) #histogram
plt.boxplot(emp.Salary_hike) #boxplot
#Graphical Representation of a input(x=Churn_out_rate)
plt.bar(height = df.Churn_out_rate, x = np.arange(len(emp)))
plt.hist(emp.Churn_out_rate) #histogram
plt.boxplot(emp.Churn_out_rate) #boxplot

# Scatter plot
plt.scatter(x = emp['Churn_out_rate'], y = emp['Salary_hike'], color = 'green') 

# correlation
np.corrcoef(emp.Churn_out_rate,emp.Salary_hike) 

# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance matrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output = np.cov(emp.Churn_out_rate, emp.Salary_hike)[0, 1]
cov_output

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('Churn_out_rate ~Salary_hike', data = emp).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(emp['Salary_hike']))

# Regression Line
plt.scatter(emp.Salary_hike, emp.Churn_out_rate)
plt.plot(emp.Salary_hike, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = emp.Churn_out_rate - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(emp['Churn_out_rate']), y = emp['Salary_hike'], color = 'brown')
np.corrcoef(np.log(emp.Salary_hike), emp.Churn_out_rate) #correlation

model2 = smf.ols('Churn_out_rate ~ np.log(Salary_hike)', data = emp).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(emp['Salary_hike']))

# Regression Line
plt.scatter(np.log(emp.Salary_hike), emp.Churn_out_rate)
plt.plot(np.log(emp.Salary_hike), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = emp.Churn_out_rate - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = emp['Churn_out_rate'], y = np.log(emp['Salary_hike']), color = 'orange')
np.corrcoef(emp.Churn_out_rate, np.log(emp.Salary_hike)) #correlation

model3 = smf.ols('np.log(Churn_out_rate) ~ Salary_hike', data =emp).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(emp['Salary_hike']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(emp.Salary_hike, np.log(emp.Churn_out_rate))
plt.plot(emp.Salary_hike, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = emp.Churn_out_rate - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(Churn_out_rate) ~ Salary_hike + I(Salary_hike*Salary_hike)', data = emp).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(emp))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = emp.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = wcat.iloc[:, 1].values


plt.scatter(emp.Salary_hike , np.log(emp.Churn_out_rate))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = emp.Churn_out_rate - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(emp, test_size = 0.2)

finalmodel = smf.ols('np.log(Churn_out_rate) ~Salary_hike + I(Salary_hike *Salary_hike)', data =train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_AT = np.exp(test_pred)
pred_test_AT

# Model Evaluation on Test data
test_res = test.Churn_out_rate - pred_test_AT
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_AT = np.exp(train_pred)
pred_train_AT

# Model Evaluation on train data
train_res = train.Churn_out_rate - pred_train_AT
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse
