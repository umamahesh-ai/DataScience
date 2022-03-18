
rn.model_selection import GridSearchCV

#Importing Data
a = pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/Logestic Regression/Datasets_LR/Affairs.csv")


a.head()
a.describe()
a.isna().sum()
a.info()

#Dropping Unnamed:0 feature.
a=a.drop("Unnamed: 0", axis=1)
a=a.drop("slghtrel", axis=1)
a=a.drop("smerel", axis=1)
a=a.drop("vryrel", axis=1)
a=a.drop("yrsmarr1", axis=1)
a=a.drop("yrsmarr2", axis=1)
a=a.drop("yrsmarr3", axis=1)
a=a.drop("yrsmarr4", axis=1)
a=a.drop("yrsmarr5", axis=1)


a.columns = a.columns.str.replace("naffairs", "A")
a.columns = a.columns.str.replace("kids", "B")
a.columns = a.columns.str.replace("vryunhap", "C")
a.columns = a.columns.str.replace("unhap", "D")
a.columns = a.columns.str.replace("avgmarr", "E")
a.columns = a.columns.str.replace("hapavg", "F")
a.columns = a.columns.str.replace("vryhap", "G")
a.columns = a.columns.str.replace("antirel", "H")
a.columns = a.columns.str.replace("notrel", "I")
a.columns = a.columns.str.replace("yrsmarr6", "J")

# Model building 
# import statsmodels.formula.api as sm
logit_model = sm.logit("J ~ A + B + C + D + E + F + G + H + I" ,data = a).fit()


#summary
logit_model.summary2() # for AIC
logit_model.summary()
a= a.iloc[:, [9,0,1,2,3,4,5,6,7,8]]
pred = logit_model.predict(a.iloc[ :, 1: ])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(a.J, pred)
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
a["pred"] = np.zeros(601)
# taking threshold value and above the prob value will be treated as correct value 
a.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(a["pred"], a["J"])
classification


### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(a, test_size = 0.3) # 30% test data

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('J ~ A + B + C + D + E + F + G + H + I', data = train_data).fit()


#summary
model.summary2() # for AIC
model.summary()

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of Clicked_on_a
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(181)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# parameter grid
parameters = {
    'penalty' : ['l1','l2'], 
    'C'       : np.logspace(-3,3,7),
    'solver'  : ['newton-cg', 'lbfgs', 'liblinear'],
}

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['J'])
confusion_matrix

accuracy_test = (46 + 56)/(181)
accuracy_test

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["J"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["J"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test


# prediction on train data
train_pred = model.predict(train_data.iloc[ :, 1: ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(420)


# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['J'])
confusion_matrx

accuracy_train = (140 + 134)/(420)
print(accuracy_train)

