import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import matplotlib
import re
from sklearn import linear_model

train_data = pd.read_csv("./train.csv")
#print train_data.info()
#print train_data.describe()
#print train_data.Cabin.value_counts()

#x = [train_data[(train_data.Sex == 'male')]['Sex'].size,train_data[(train_data.Sex == 'female')]['Sex'].size]
#y = [train_data[(train_data.Sex == 'male') & (train_data.Survived == 1)]['Sex'].size,train_data[(train_data.Sex == 'female') & (train_data.Survived == 1)]['Sex'].size]

#print x
#print y

train_data.Embarked[train_data.Embarked.isnull()] = train_data.Embarked.dropna().mode().values
train_data.Cabin[train_data.Cabin.isnull()] = 'U0'
age_df = train_data[['Age','Survived','Fare','Parch','SibSp','Pclass']]
#print age_df

age_df_notnull = age_df.loc[(train_data.Age.notnull())]
age_df_isnull = age_df.loc[(train_data.Age.isnull())]

#print age_df_notnull
#print age_df_isnull

X = age_df_notnull.values[:,1:]
Y = age_df_notnull.values[:,0]

#print X
#print Y

rfr = RandomForestRegressor(n_estimators = 1000,n_jobs = -1)
rfr.fit(X,Y)
predictages = rfr.predict(age_df_isnull.values[:,1:])
train_data.loc[(train_data.Age.isnull()),'Age'] = predictages
#print train_data.info()

dummy_df = pd.get_dummies(train_data.Embarked)
#print dummy_df
dummy_df = dummy_df.rename(columns = lambda x:'Embarked_' + str(x))
#print dummy_df
train_data = pd.concat([train_data,dummy_df],axis = 1)

#print train_data.info()

def getCabinNumber(cabin):
	match = re.compile("([0-9]+)").search(cabin)
	if match:
		return match.group()
	else:
		return 0

def getSex(sex):
	if sex == 'male':
		return 1
	else :
		return 0

train_data['Cabinletter'] = train_data['Cabin'].map(lambda x : re.compile("([a-zA-Z]+)").search(x).group())
train_data['Cabinletter'] =	pd.factorize(train_data.Cabinletter)[0]
train_data['Cabinnumber'] = train_data['Cabin'].map(lambda x : getCabinNumber(x)).astype(int) + 1
train_data['Sex'] = train_data['Sex'].map(lambda x : getSex(x))
#print train_data['Sex']


scaler = preprocessing.StandardScaler()
age_scaler_param = scaler.fit(train_data['Age'])
train_data['Age_scaled'] = scaler.fit_transform(train_data['Age'],age_scaler_param)
fare_scale_param = scaler.fit(train_data['Fare'])
train_data['Fare_scaled'] = scaler.fit_transform(train_data['Fare'],fare_scale_param)

#print train_data
train_np = train_data.filter(regex = 'Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex|Pclass')
train_np = train_np.as_matrix()
#print train_np

y = train_np[:,0]
x = train_np[:,1:]

logistic = linear_model.LogisticRegression(C = 1.0 , penalty = 'l1' , tol = 1e-6)
logistic.fit(x,y)
print logistic




'''
handle test data
'''

test_data = pd.read_csv("./test.csv")
print test_data.info()

test_data.Fare[test_data.Fare.isnull()] = test_data.Fare.mean()
#print test_data.info()


test_data.Embarked[test_data.Embarked.isnull()] = test_data.Embarked.dropna().mode().values
test_data.Cabin[test_data.Cabin.isnull()] = 'U0'
test_age_df = test_data[['Age','Fare','Parch','SibSp','Pclass']]
#print age_df

test_age_df_notnull = test_age_df.loc[(test_data.Age.notnull())]
test_age_df_isnull = test_age_df.loc[(test_data.Age.isnull())]

#print age_df_notnull
#print age_df_isnull

test_X = test_age_df_notnull.values[:,1:]
test_Y = test_age_df_notnull.values[:,0]

#print test_data
#print X
#print Y
test_rfr = RandomForestRegressor(n_estimators = 1000,n_jobs = -1)
test_rfr.fit(test_X,test_Y)
test_predictages = test_rfr.predict(test_age_df_isnull.values[:,1:])
test_data.loc[(test_data.Age.isnull()),'Age'] = test_predictages

#print train_data.info()

test_dummy_df = pd.get_dummies(test_data.Embarked)
#print dummy_df
test_dummy_df = test_dummy_df.rename(columns = lambda x:'Embarked_' + str(x))
#print dummy_df
test_data = pd.concat([test_data,test_dummy_df],axis = 1)

test_data['Cabinletter'] = test_data['Cabin'].map(lambda x : re.compile("([a-zA-Z]+)").search(x).group())
test_data['Cabinletter'] =	pd.factorize(test_data.Cabinletter)[0]
test_data['Cabinnumber'] = test_data['Cabin'].map(lambda x : getCabinNumber(x)).astype(int) + 1
test_data['Sex'] = test_data['Sex'].map(lambda x : getSex(x))
#print train_data['Sex']


test_scaler = preprocessing.StandardScaler()
test_age_scaler_param = test_scaler.fit(test_data['Age'])
test_data['Age_scaled'] = test_scaler.fit_transform(test_data['Age'],test_age_scaler_param)
test_fare_scale_param = test_scaler.fit(test_data['Fare'])
test_data['Fare_scaled'] = test_scaler.fit_transform(test_data['Fare'],test_fare_scale_param)

#print train_data
test_np = test_data.filter(regex = 'Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex|Pclass')
predictresult = logistic.predict(test_np)
result = pd.DataFrame({'PassengerId':test_data['PassengerId'].as_matrix(), 'Survived':predictresult.astype(np.int32)})
result.to_csv("./result.csv", index=False)
#print test_np

















