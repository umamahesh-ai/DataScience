
# Multilinear Regression
import pandas as pd
import numpy as np

# loading the data
start = pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/multi linear regression/Datasets_MLR/50_Startups.csv")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

start.describe()
start.info()
#converting categorical to numerical data
from sklearn.preprocessing import LabelEncoder
la=LabelEncoder()
start['State']=la.fit_transform(start['State'])
# Rename the columns t: t_new
start1 = start.rename(columns = {'R&D Spend': 'RD', 'Marketing Spend': 'MD'}, inplace = False)

# Print out start1
print(start1)


#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# Profit
plt.bar(height = start1.Profit, x = np.arange(1, 82, 1))
plt.hist(start1.Profit) #histogram
plt.boxplot(start1.Profit) #boxplot


# Jointplot
import seaborn as sns
sns.jointplot(x=start1['Profit'], y=start1['MD'])

# Q-Q Plot/ to check data is normally distributed or not
from scipy import stats
import pylab
stats.probplot(start1.Profit, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(start1.iloc[:, :])
                             
# Correlation matrix 
start1.corr()


# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('Profit ~ RD + Administration + MD + State', data = start1).fit() # regression model

# Summary
ml1.summary()
# p-values for Administration, State are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 49 is showing high influence so we can exclude that entire row

start1_new = start1.drop(start1.index[[49]])

# Preparing model                  
ml_new = smf.ols('Profit ~ RD + Administration + MD + State', data = start1_new).fit()     

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_RD = smf.ols('RD ~ Administration + MD + State''', data = start1).fit().rsquared  
vif_RD = 1/(1 - rsq_RD) 

rsq_Administration = smf.ols('Administration ~ MD + State + RD', data = start1).fit().rsquared  
vif_Administration = 1/(1 - rsq_Administration)

rsq_MD = smf.ols('MD ~ Administration + State + RD', data = start1).fit().rsquared  
vif_MD = 1/(1 - rsq_MD) 

rsq_State = smf.ols('State ~ Administration + RD + MD', data = start1).fit().rsquared  
vif_State = 1/(1 - rsq_State) 

# Storing vif values in a data frame
d1 = {'Variables':['RD', 'Administration', 'MD', 'State'], 'VIF':[vif_RD, vif_Administration, vif_MD, vif_State]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As RD is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('Profit ~  Administration + MD + State', data = start1).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(start1)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = start1.Profit, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
start1_train, start1_test = train_test_split(start1, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("Profit ~  Administration + MD + State", data = start1_train).fit()

# prediction on test data set 
test_pred = model_train.predict(start1_test)

# test residual values 
test_resid = test_pred - start1_test.Profit
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(start1_train)

# train residual values 
train_resid  = train_pred - start1_train.Profit
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
