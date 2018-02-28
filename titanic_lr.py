import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn import linear_model
import sklearn
import re
import tensorflow as tf 
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score

train_data = pd.read_csv("./train.csv")


def getlenofname(x):
	return len(x)

def getnameprefix(x):
	return x.split(',')[1].split('.')[0]


def getsex(x):
	if x == 'male':
		return 1
	else:
		return 0

def getCabinNumber(cabin):
	match = re.compile("([0-9]+)").search(cabin)
	if match:
		return match.group()
	else:
		return 0



train_data["Sex"] = train_data["Sex"].map(lambda x:getsex(x)).astype(int)
train_data["Name_len"] = train_data["Name"].map(lambda x:getlenofname(x)).astype(int)
train_data["Name_prefix"] = train_data["Name"].map(lambda x:getnameprefix(x)).astype(str)
 
#train_data.Cabin[train_data.Cabin.isnull()] = "UO"
train_data.Cabin[train_data.Cabin.isnull()] = 'U0'

#print train_data.head(5)
scaler = preprocessing.StandardScaler()
fare_scale_param = scaler.fit(train_data['Fare'])
train_data['Fare_scaled'] = scaler.fit_transform(train_data['Fare'],fare_scale_param)

max_fare = train_data["Fare"].max()
min_fare = train_data["Fare"].min()
mean_fare = train_data["Fare"].mean()


train_data.Embarked[train_data.Embarked.isnull()] = train_data.Embarked.dropna().mode().values
dummy_name_prefix = pd.get_dummies(train_data["Name_prefix"],prefix = "prename")
train_data = pd.concat([train_data,dummy_name_prefix],axis = 1)


dummy_df = pd.get_dummies(train_data.Embarked)
dummy_df = dummy_df.rename(columns = lambda x:'Embarked_' + str(x))
train_data = pd.concat([train_data,dummy_df],axis = 1)

train_data['Cabin_letter'] = train_data['Cabin'].map(lambda x : re.compile("([a-zA-Z]+)").search(x).group())
train_data['Cabin_letter'] = pd.factorize(train_data.Cabin_letter)[0]
train_data['Cabin_number'] = train_data['Cabin'].map(lambda x : getCabinNumber(x)).astype(int) + 1

train_data['home_num'] = train_data['SibSp'] + train_data['Parch'] + 1

age_df = train_data[["Age","Survived","Pclass","SibSp","Parch","Fare_scaled","Name_len"]]
age_df_notnull = age_df.loc[(train_data.Age.notnull())]
age_df_isnull = age_df.loc[(train_data.Age.isnull())]

X = age_df_notnull.values[:,1:]
Y = age_df_notnull.values[:,0]

rfp = RandomForestRegressor(n_estimators = 3000,n_jobs = -1)
rfp.fit(X,Y)
predict_age = rfp.predict(age_df_isnull.values[:,1:])
train_data.loc[train_data["Age"].isnull(),"Age"] = predict_age
age_scale_param = scaler.fit(train_data["Age"])
train_data["Age_scaled"] = scaler.fit_transform(train_data["Age"],age_scale_param)



test_data = pd.read_csv("./test.csv")

test_data.Fare[test_data.Fare.isnull()] = test_data.Fare.mean()
test_data["Sex"] = test_data["Sex"].map(lambda x:getsex(x)).astype(int)
test_data["Name_len"] = test_data["Name"].map(lambda x:getlenofname(x)).astype(int)
test_data["Name_prefix"] = test_data["Name"].map(lambda x:getnameprefix(x)).astype(str)
 
#test_data.Cabin[test_data.Cabin.isnull()] = "UO"
test_data.Cabin[test_data.Cabin.isnull()] = 'U0'

#print test_data.head(5)
scaler = preprocessing.StandardScaler()
fare_scale_param = scaler.fit(test_data['Fare'])
test_data['Fare_scaled'] = scaler.fit_transform(test_data['Fare'],fare_scale_param)

max_fare = test_data["Fare"].max()
min_fare = test_data["Fare"].min()
mean_fare = test_data["Fare"].mean()



test_data.Embarked[test_data.Embarked.isnull()] = test_data.Embarked.dropna().mode().values
dummy_name_prefix = pd.get_dummies(test_data["Name_prefix"],prefix = "prename")
test_data = pd.concat([test_data,dummy_name_prefix],axis = 1)


dummy_df = pd.get_dummies(test_data.Embarked)
dummy_df = dummy_df.rename(columns = lambda x:'Embarked_' + str(x))
test_data = pd.concat([test_data,dummy_df],axis = 1)

test_data['Cabin_letter'] = test_data['Cabin'].map(lambda x : re.compile("([a-zA-Z]+)").search(x).group())
test_data['Cabin_letter'] = pd.factorize(test_data.Cabin_letter)[0]
test_data['Cabin_number'] = test_data['Cabin'].map(lambda x : getCabinNumber(x)).astype(int) + 1

test_data['home_num'] = test_data['SibSp'] + test_data['Parch'] + 1

age_df = test_data[["Age","Pclass","SibSp","Parch","Fare_scaled","Name_len"]]
age_df_notnull = age_df.loc[(test_data.Age.notnull())]
age_df_isnull = age_df.loc[(test_data.Age.isnull())]

X = age_df_notnull.values[:,1:]
Y = age_df_notnull.values[:,0]

rfp = RandomForestRegressor(n_estimators = 3000,n_jobs = -1)
rfp.fit(X,Y)
predict_age = rfp.predict(age_df_isnull.values[:,1:])
test_data.loc[test_data["Age"].isnull(),"Age"] = predict_age
age_scale_param = scaler.fit(test_data["Age"])
test_data["Age_scaled"] = scaler.fit_transform(test_data["Age"],age_scale_param)


train_pre = train_data.filter(regex = 'Survived|Pclass|Sex|Age_scaled|Name_len|SibSp|Parch|Fare_scaled|Embarked_.*|home_num')
test_pre = test_data.filter(regex = 'Pclass|Sex|Age_scaled|Name_len|SibSp|Parch|Fare_scaled|Embarked_.*|home_num')



#for i in train_pre.columns:
#	if i not in test_pre.columns and i != 'Survived':
#		test_pre[i] = 0

#for i in test_pre.columns:
#	if i not in train_pre.columns:
#		train_pre[i] = 0

print train_pre.info()
print test_pre.info()

train_pre = train_pre.as_matrix()
y = train_pre[:,0]
x = train_pre[:,1:]

#regressor_model = linear_model.LogisticRegression(C = 1.0 , penalty = 'l2' , tol = 1e-8)
#model = regressor_model.fit(x,y)
#print model

test_pre = test_pre.as_matrix()
#clf =XGBClassifier(learning_rate=0.1, n_estimators = 32, max_depth=5, silent=True, objective='binary:logistic').fit(x,y)

#predictresult = clf.predict(test_pre)
logistic = linear_model.LogisticRegression(C = 1.0 , penalty = 'l1' , tol = 1e-6).fit(x,y)


predictresult = logistic.predict(test_pre)
result = pd.DataFrame({'PassengerId':test_data['PassengerId'].as_matrix(), 'Survived':predictresult.astype(np.int32)})
result.to_csv("./result.csv", index=False)

