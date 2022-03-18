

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

#Importing Data
ad = pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/Logestic Regression/Datasets_LR/advertising.csv")


ad.head()
ad.describe()
ad.isna().sum()
ad.info()
#converting discrete data to int
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
ad["City"]=lb.fit_transform(ad["City"])
ad['Ad_Topic_Line']=lb.fit_transform(ad['Ad_Topic_Line'])
ad["Country"]=lb.fit_transform(ad["Country"])

#Dropping Timestamp feature.
ad=ad.drop("Timestamp", axis=1)

help(ad.astype)
# Now we will convert 'float64' into 'int64' type. typecasting done for float
ad.Daily_Time_Spent_on_Site =ad.Daily_Time_Spent_on_Site.astype("int64")
ad.Area_Income =ad.Area_Income.astype("int64")
ad.Daily_Internet_Usage =ad.Daily_Internet_Usage.astype("int64")
ad= ad.iloc[:, [8,0,1,2,3,4,5,6,7]]
# Model building 
# import statsmodels.formula.api as sm
logit_model = sm.logit('Clicked_on_Ad ~ Daily_Time_Spent_on_Site+ Age + Area_Income+ Daily_Internet_Usage+ Ad_Topic_Line+ City+Male+ Country', data = ad).fit()

#summary
logit_model.summary2() # for AIC
logit_model.summary()

pred = logit_model.predict(ad.iloc[ :, 1: ])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(ad.Clicked_on_Ad, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

# filling all the cells with zeroes
ad["pred"] = np.zeros(1000)
# taking threshold value and above the prob value will be treated as correct value 
ad.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(ad["pred"], ad["Clicked_on_Ad"])
classification


### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(ad, test_size = 0.3) # 30% test data

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('Clicked_on_Ad ~ Daily_Time_Spent_on_Site+ Age + Area_Income+ Daily_Internet_Usage+ Ad_Topic_Line+ City+Male+ Country', data = train_data).fit()


#summary
model.summary2() # for AIC
model.summary()

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of Clicked_on_Ad
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(300)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['Clicked_on_Ad'])
confusion_matrix

accuracy_test = (151 + 141)/(300)
accuracy_test

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["Clicked_on_Ad"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["Clicked_on_Ad"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test


# prediction on train data
train_pred = model.predict(train_data.iloc[ :, 1: ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(700)


# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['Clicked_on_Ad'])
confusion_matrx

accuracy_train = (334 + 349)/(700)
print(accuracy_train)
