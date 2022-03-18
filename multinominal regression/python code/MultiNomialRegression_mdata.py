
#import libraries
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#loading the dataset
mdata = pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/Multinominal Regression/Datasets_Multinomial/mdata.csv")
mdata.head()
#label encoder
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
mdata['female'] = lb.fit_transform(mdata['female'])
mdata['ses'] = lb.fit_transform(mdata['ses'])
mdata['schtyp'] = lb.fit_transform(mdata['schtyp'])
mdata['prog'] = lb.fit_transform(mdata['prog'])
mdata['honors'] = lb.fit_transform(mdata['honors'])
mdata['read'] = lb.fit_transform(mdata['read'])
mdata['write'] = lb.fit_transform(mdata['write'])
mdata['math'] = lb.fit_transform(mdata['math'])
mdata['science'] = lb.fit_transform(mdata['science'])
mdata
mdata.ses.value_counts()
#boxplot
sns.boxplot(x = "ses", y = "read", data = mdata)
sns.boxplot(x = "ses", y = "write", data = mdata)
sns.boxplot(x = "ses", y = "math", data = mdata)
sns.boxplot(x = "ses", y = "science", data = mdata)
#stripplot
sns.stripplot(x = "ses", y = "schtyp", jitter = True, data = mdata)
sns.stripplot(x = "ses", y = "prog", jitter = True, data = mdata)
sns.stripplot(x = "ses", y = "read", jitter = True, data = mdata)
sns.stripplot(x = "ses", y = "write", jitter = True, data = mdata)
sns.stripplot(x = "ses", y = "math", jitter = True, data = mdata)
sns.stripplot(x = "ses", y = "science", jitter = True, data = mdata)
sns.stripplot(x = "ses", y = "honors", jitter = True, data = mdata)
#pairplot
sns.pairplot(mdata)
sns.pairplot(mdata, hue = "ses") 
#corr
mdata.corr()
#train test split
train, test = train_test_split(mdata, test_size = 0.2)
mdata = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train.iloc[:, 1:], train.iloc[:, 1])
test_predict = mdata.predict(test.iloc[:, 1:]) # Test predictions
#accuracy 
accuracy_score(test.iloc[:,0], test_predict)
train_predict = mdata.predict(train.iloc[:, 1:]) # Train predictions 
accuracy_score(train.iloc[:,0], train_predict) 
