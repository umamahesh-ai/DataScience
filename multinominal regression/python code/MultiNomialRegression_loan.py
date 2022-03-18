

### Multinomial Regression ####
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#loading the data
loan = pd.read_csv("C:/Users/HAI/Desktop/360DigitMG/Assingment/Multinominal Regression/Datasets_Multinomial/loan.csv")
loan.head()
loan.isna().sum()
loan=loan.dropna(axis=1)
loan.duplicated().sum()
loan.describe()
loan.info()
loan=loan.iloc[:,2:]
loan=loan.drop(['issue_d','url','zip_code','addr_state','earliest_cr_line','initial_list_status','policy_code','acc_now_delinq','delinq_amnt','application_type','pub_rec','pymnt_plan','total_rec_late_fee','recoveries','out_prncp_inv','out_prncp','delinq_2yrs','collection_recovery_fee'],axis=1)
#labelbncoding
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
loan['term']=lb.fit_transform(loan['term'])
loan['int_rate']=lb.fit_transform(loan['int_rate'])
loan['grade']=lb.fit_transform(loan['grade'])
loan['sub_grade']=lb.fit_transform(loan['sub_grade'])
loan['home_ownership']=lb.fit_transform(loan['home_ownership'])
loan['verification_status']=lb.fit_transform(loan['verification_status'])
loan['purpose']=lb.fit_transform(loan['purpose'])
loan.term.value_counts()
#boxplot
sns.boxplot(x = "term", y = "funded_amnt", data = loan)
sns.boxplot(x = "term", y = "funded_amnt_inv", data = loan)
sns.boxplot(x = "term", y = "term", data = loan)
sns.boxplot(x = "term", y = "int_rate", data = loan)
sns.boxplot(x = "term", y = "grade", data = loan)
sns.boxplot(x = "term", y = "sub_grade", data = loan)
sns.boxplot(x = "term", y = "home_ownership", data = loan)
#stripplot
sns.stripplot(x = "term", y = "funded_amnt", jitter = True, data = loan)
sns.stripplot(x = "term", y = "funded_amnt_inv", jitter = True, data = loan)
sns.stripplot(x = "term", y = "term", jitter = True, data = loan)
sns.stripplot(x = "term", y = "int_rate", jitter = True, data = loan)
sns.stripplot(x = "term", y = "grade", jitter = True, data = loan)
sns.stripplot(x = "term", y = "sub_grade", jitter = True, data = loan)
sns.stripplot(x = "term", y = "home_ownership", jitter = True, data = loan)
#corr
loan.corr()
#split
x=loan.drop(['loan_status'],axis=1)
y=loan['loan_status']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=LogisticRegression(multi_class='multinomial',solver='newton-cg').fit(x_train,y_train)
test_pre=model.predict(x_test)
train_pre=model.predict(x_train)
#accuracy
accuracy_score(test_pre,y_test)
accuracy_score(train_pre,y_train)

