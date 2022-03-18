import pandas as pd
import numpy as np
Insurance=pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/Hirerical clustering/Dataset_Assignment Clustering/AutoInsurance.csv")
Insurance.describe()
Insurance.info()
# In the dataset having the some categorical variables and numerical variables
# Convert into  categorical data into numerical data
# saperate the categorical and numerical variables  by using include and exclude 
from sklearn.preprocessing import LabelEncoder
categorical=Insurance.select_dtypes(include=['object']) # include is the categorical
numerical=Insurance.select_dtypes(exclude=['object']) # exclude is the numerical
#cancatinade the categorical and numerical with axis =1 means add cloumn side 
new_df=pd.concat((categorical,numerical),axis=1)
# convert the categorical to numerical
AutoInsurance=new_df.loc[:,'Customer':'Vehicle Size']=new_df.loc[:,'Customer':'Vehicle Size'].apply( LabelEncoder().fit_transform)
# Normalization function
def norm(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
data=norm(AutoInsurance)








