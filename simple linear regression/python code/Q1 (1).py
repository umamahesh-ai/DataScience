import pandas as pd
import numpy as np
df=pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/Simple linear regresion/Datasets_SLR/delivery_time.csv")
# Exploratory data analysis:
# 1. Measures of central tendency
####calculate the mean 
df.Delivery_Time.mean()
df.Sorting_Time.mean()
####calculate the median
df.Delivery_Time.median()
df.Sorting_Time.median()
# 2. Measures of dispersion
#calculate the std
df.Delivery_Time.std()
df.Sorting_Time.std()
#calculate variance
df.Delivery_Time.var()
df.Sorting_Time.var()
#calculate the Range
range_output=max(df.Delivery_Time)-min(df.Delivery_Time)
range_input=max(df.Sorting_Time)-min(df.Sorting_Time)
# 3. Third moment business decision
df.Delivery_Time.skew()
df.Sorting_Time.skew()
# 4. Fourth moment business decision
df.Delivery_Time.kurt()
df.Sorting_Time.kurt()
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, 
df.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 
#Graphical Representation of a output(y=Delivery_Time)
plt.bar(height = df.Delivery_Time, x = np.arange(len(df)))
plt.hist(df.Delivery_Time) #histogram
plt.boxplot(df.Delivery_Time) #boxplot
#Graphical Representation of a input(x=Sorting_Time)
plt.bar(height = df.Sorting_Time, x = np.arange(len(df)))
plt.hist(df.Sorting_Time) #histogram
plt.boxplot(df.Sorting_Time) #boxplot

# Scatter plot
plt.scatter(x = df['Sorting_Time'], y = df['Delivery_Time'], color = 'green') 

# correlation
np.corrcoef(df.Delivery_Time,df.Sorting_Time) 

# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance matrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output = np.cov(df.Delivery_Time, df.Sorting_Time)[0, 1]
cov_output

# wcat.cov()


# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('Sorting_Time ~Delivery_Time', data = df).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(df['Delivery_Time']))

# Regression Line
plt.scatter(df.Delivery_Time, df.Sorting_Time)
plt.plot(df.Delivery_Time, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = df.Sorting_Time - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(df['Delivery_Time']), y = df['Sorting_Time'], color = 'brown')
np.corrcoef(np.log(df.Delivery_Time), df.Sorting_Time) #correlation

model2 = smf.ols('Sorting_Time ~ np.log(Delivery_Time)', data = df).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(df['Delivery_Time']))

# Regression Line
plt.scatter(np.log(df.Delivery_Time), df.Sorting_Time)
plt.plot(np.log(df.Delivery_Time), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = df.Sorting_Time - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = df['Delivery_Time'], y = np.log(df['Sorting_Time']), color = 'orange')
np.corrcoef(df.Delivery_Time, np.log(df.Sorting_Time)) #correlation

model3 = smf.ols('np.log(Sorting_Time) ~ Delivery_Time', data =df).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(df['Delivery_Time']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(df.Delivery_Time, np.log(df.Sorting_Time))
plt.plot(df.Delivery_Time, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = df.Sorting_Time - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(Sorting_Time) ~ Delivery_Time + I(Delivery_Time*Delivery_Time)', data = df).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(df))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = df.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = wcat.iloc[:, 1].values


plt.scatter(df.Delivery_Time, np.log(df.Sorting_Time))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = df.Sorting_Time - pred4_at
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

train, test = train_test_split(df, test_size = 0.2)

finalmodel = smf.ols('np.log(Sorting_Time) ~ Delivery_Time + I(Delivery_Time * Delivery_Time)', data =train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_AT = np.exp(test_pred)
pred_test_AT

# Model Evaluation on Test data
test_res = test.Sorting_Time - pred_test_AT
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_AT = np.exp(train_pred)
pred_train_AT

# Model Evaluation on train data
train_res = train.Sorting_Time - pred_train_AT
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse
