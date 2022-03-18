import pandas as pd
import numpy as np
df=pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/Simple linear regresion/Datasets_SLR/SAT_GPA.csv")
df.columns
# Exploratory data analysis:
# 1. Measures of central tendency
####calculate the mean 
df.SAT_Scores.mean()
df.GPA.mean()
####calculate the median
df.SAT_Scores.median()
df.GPA.median()
# 2. Measures of dispersion
#calculate the std
df.SAT_Scores.std()
df.GPA.std()
#calculate variance
df.SAT_Scores.var()
df.GPA.var()
#calculate the Range
range_output=max(df.SAT_Scores)-min(df.SAT_Scores)
range_input=max(df.GPA)-min(df.GPA)
# 3. Third moment business decision
df.SAT_Scores.skew()
df.GPA.skew()
# 4. Fourth moment business decision
df.SAT_Scores.kurt()
df.GPA.kurt()
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, 
df.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 
#Graphical Representation of a output(y=SAT_Scores)
plt.bar(height = df.SAT_Scores, x = np.arange(len(df)))
plt.hist(df.SAT_Scores) #histogram
plt.boxplot(df.SAT_Scores) #boxplot
#Graphical Representation of a input(x=GPA)
plt.bar(height = df.GPA, x = np.arange(len(df)))
plt.hist(df.GPA) #histogram
plt.boxplot(df.GPA) #boxplot

# Scatter plot
plt.scatter(x = df['GPA'], y = df['SAT_Scores'], color = 'green') 

# correlation
np.corrcoef(df.SAT_Scores,df.GPA) 

# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance matrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output = np.cov(df.SAT_Scores, df.GPA)[0, 1]
cov_output

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('GPA ~SAT_Scores', data = df).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(df['SAT_Scores']))

# Regression Line
plt.scatter(df.SAT_Scores, df.GPA)
plt.plot(df.SAT_Scores, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = df.GPA - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(df['SAT_Scores']), y = df['GPA'], color = 'brown')
np.corrcoef(np.log(df.SAT_Scores), df.GPA) #correlation

model2 = smf.ols('GPA ~ np.log(SAT_Scores)', data = df).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(df['SAT_Scores']))

# Regression Line
plt.scatter(np.log(df.SAT_Scores), df.GPA)
plt.plot(np.log(df.SAT_Scores), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = df.GPA - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = df['SAT_Scores'], y = np.log(df['GPA']), color = 'orange')
np.corrcoef(df.SAT_Scores, np.log(df.GPA)) #correlation

model3 = smf.ols('np.log(GPA) ~ SAT_Scores', data =df).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(df['SAT_Scores']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(df.SAT_Scores, np.log(df.GPA))
plt.plot(df.SAT_Scores, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = df.GPA - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(GPA) ~ SAT_Scores + I(SAT_Scores*SAT_Scores)', data = df).fit()
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


plt.scatter(df.SAT_Scores, np.log(df.GPA))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = df.GPA - pred4_at
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

finalmodel = smf.ols('np.log(GPA) ~SAT_Scores + I(SAT_Scores * SAT_Scores)', data =train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_AT = np.exp(test_pred)
pred_test_AT

# Model Evaluation on Test data
test_res = test.GPA - pred_test_AT
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_AT = np.exp(train_pred)
pred_train_AT

# Model Evaluation on train data
train_res = train.GPA - pred_train_AT
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse
