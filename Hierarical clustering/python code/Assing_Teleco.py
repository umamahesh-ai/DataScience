import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_excel("C:/Users/HAI/Desktop/360DigitMG/Assingment/Hirerical clustering/Dataset_Assignment Clustering/Telco_customer_churn.xlsx")

df.describe()
df.info()
# In the dataset having the some categorical variables and numerical variables
# Convert into  categorical data into numerical data
# saperate the categorical and numerical variables  by using include and exclude 
from sklearn.preprocessing import LabelEncoder
categorical=df.select_dtypes(include=['object']) # include is the categorical
numerical=df.select_dtypes(exclude=['object']) # exclude is the numerical
#cancatinade the categorical and numerical with axis =1 means add cloumn side 
new_df=pd.concat((categorical,numerical),axis=1)
# convert the categorical to numerical
Teleco_customer=new_df.loc[:,'Customer ID':'Payment Method']=new_df.loc[:,'Customer ID':'Payment Method'].apply( LabelEncoder().fit_transform)



