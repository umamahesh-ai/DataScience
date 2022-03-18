import pandas as pd
import numpy as np
calories=pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/Simple linear regresion/Datasets_SLR/calories_consumed.csv")
calories.columns
# Exploratory data analysis:
# 1. Measures of central tendency
####calculate the mean 
calories.Weight_gained.mean()
calories.Weight_calories.mean()
####calculate the median
calories.Weight_gained.median()
calories.Weight_calories.median()
# 2. Measures of dispersion
#calculate the std
calories.Weight_gained.std()
calories.Weight_calories.std()
#calculate variance
calories.Weight_gained.var()
calories.Weight_calories.var()
#calculate the Range
range_output=max(calories.Weight_gained)-min(calories.Weight_gained)
range_input=max(calories.Weight_calories)-min(calories.Weight_calories)
# 3. Third moment business decision
calories.Weight_gained.skew()
calories.Weight_calories.skew()
# 4. Fourth moment business decision
calories.Weight_gained.kurt()
calories.Weight_calorieskurt()
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, 
calories.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 
#Graphical Representation of a output(y=Weight_gained)
plt.bar(height = calories.Weight_gained, x = np.arange(len(calories)))
plt.hist(calories.Weight_gained) #histogram
plt.boxplot(calories.Weight_gained) #boxplot
#Graphical Representation of a input(x=Weight_calories)
plt.bar(height = calories.Weight_calories, x = np.arange(len(calories)))
plt.hist(calories.Weight_calories) #histogram
plt.boxplot(calories.Weight_calories) #boxplot

# Scatter plot
plt.scatter(x = calories['Weight_calories'], y = calories['Weight_gained'], color = 'green') 

# correlation
np.corrcoef(calories.Weight_gained,calories.Weight_calories) 

# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance matrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output = np.cov(calories.Weight_gained,calories.Weight_calories)[0, 1]
cov_output

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('Weight_calories ~Weight_gained', data = calories).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(calories['Weight_gained']))

# Regression Line
plt.scatter(calories.Weight_calories,calories.Weight_gained)
plt.plot(calories.Weight_gained, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = calories.Weight_calories - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(calories['Weight_calories']), y = calories['Weight_gained'], color = 'brown')
np.corrcoef(np.log(calories.Weight_gained), calories.Weight_calories) #correlation

model2 = smf.ols('Weight_calories ~ np.log(Weight_gained)', data = calories).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(calories['Weight_gained']))

# Regression Line
plt.scatter(np.log(calories.Weight_gained), calories.Weight_calories)
plt.plot(np.log(calories.Weight_gained), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = calories.Weight_calories - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = calories['Weight_calories'], y = np.log(calories['Weight_gained']), color = 'orange')
np.corrcoef(calories.Weight_gained, np.log(calories.Weight_calories)) #correlation

model3 = smf.ols('np.log(Weight_calories) ~ Weight_gained', data =calories).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(calories['Weight_gained']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(calories.Weight_gained, np.log(calories.Weight_calories))
plt.plot(calories.Weight_gained, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = calories.Weight_calories - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(Weight_calories) ~ Weight_gained + I(Weight_gained*Weight_gained)', data = calories).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(calories))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = calories.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = wcat.iloc[:, 1].values


plt.scatter(calories.Weight_gained, np.log(calories.Weight_calories))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = calories.Weight_calories - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(calories)
table_rmse

###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(calories, test_size = 0.2)

finalmodel = smf.ols('np.log(Weight_calories) ~ Weight_gained + I(Weight_gained * Weight_gained)', data =train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_AT = np.exp(test_pred)
pred_test_AT

# Model Evaluation on Test data
test_res = test.Weight_calories - pred_test_AT
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_AT = np.exp(train_pred)
pred_train_AT

# Model Evaluation on train data
train_res = train.Weight_gained - pred_train_AT
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse
