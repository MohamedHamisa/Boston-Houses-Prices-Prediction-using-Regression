import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
%matplotlib inline
warnings.filterwarnings('ignore')

df = pd.read_csv("Boston Dataset.csv")
df.drop(columns=['Unnamed: 0'], axis=0, inplace=True)
df.head()

# statistical info
df.describe()

# datatype info
df.info()

# check for null values
df.isnull().sum()

#Exploratory Data Analysis
# create box plots
fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
ax = ax.flatten()

for col, value in df.items():
    sns.boxplot(y=col, data=df, ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)


# create dist plot
fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
ax = ax.flatten()

for col, value in df.items():
    sns.distplot(value, ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)


#Min-Max Normalization
cols = ['crim', 'zn', 'tax', 'black']
for col in cols:
    # find minimum and maximum of that column
    minimum = min(df[col])
    maximum = max(df[col])
    df[col] = (df[col] - minimum) / (maximum - minimum)

    fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
ax = ax.flatten()

for col, value in df.items():
    sns.distplot(value, ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)

# standardization
from sklearn import preprocessing
scalar = preprocessing.StandardScaler()

# fit our data
scaled_cols = scalar.fit_transform(df[cols])
scaled_cols = pd.DataFrame(scaled_cols, columns=cols)
scaled_cols.head()

for col in cols:
    df[col] = scaled_cols[col]
fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
ax = ax.flatten()

for col, value in df.items():
    sns.distplot(value, ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)

#Coorelation Matrix
corr = df.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr, annot=True, cmap='coolwarm')

sns.regplot(y=df['medv'], x=df['lstat'])

sns.regplot(y=df['medv'], x=df['rm'])

X = df.drop(columns=['medv', 'rad'], axis=1)
y = df['medv']


#Model Training
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
def train(model, X, y):
    # train the model
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model.fit(x_train, y_train)
    
    # predict the training set
    pred = model.predict(x_test)
    
    # perform cross-validation
    cv_score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    cv_score = np.abs(np.mean(cv_score))
    
    print("Model Report")
    print("MSE:",mean_squared_error(y_test, pred))
    print('CV Score:', cv_score)


from sklearn.linear_model import LinearRegression
model = LinearRegression(normalize=True)
train(model, X, y)
coef = pd.Series(model.coef_, X.columns).sort_values()
coef.plot(kind='bar', title='Model Coefficients')

from sklearn.tree import DecisionTreeRegressor  #it's supervised learning algorithms
#it's better to set the splitter parameter to random
#max and min depth : int between 1:100 is equal to the number of attributes and it equals number of nodes 
#it works on the features of the objects and train the model in form of tree to produce continous output like profit of product
#min sample split : it's required samples number to to split an internal nodes if it's float take ceil 
#min sample leaf :it's minimum samples that required to be on a leaf
#min weight fraction leaf : sample have equal weights when sample weight is not applied
#when it's applied it means the minimum wighted fraction of the sum total of weights of all input sample that required to be on a leaf node
#max features :it's considers when looking for the best split if int it's considerd at each split
#if float it equals (max_fea*n_fea)
#random state : how to split the data default = None 
#max leaf node : how many leaves in the node
#min impurity decrease : node will be split when this split makes a decrease of the impurity greater than or equal to this value
#ccp_alpha : used for minimal cost complexity the largest cost that is smaller than alpha will be chosen
model = DecisionTreeRegressor()
train(model, X, y)
coef = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title='Feature Importance')

from sklearn.ensemble import RandomForestRegressor  #prediction by voting between the trees
model = RandomForestRegressor()
train(model, X, y)
coef = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title='Feature Importance')

from sklearn.ensemble import ExtraTreesRegressor  #increasr number of estimators or trees and calc the average to get better accuracy
#random forest is alittle bit better than  extra trees because the outputs varies for 4 or 5 datasets even the values are repeated
model = ExtraTreesRegressor()
train(model, X, y)
coef = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title='Feature Importance')

import xgboost as xgb
model = xgb.XGBRegressor()   #using gradient boosting it provides parallel tree boosting 
train(model, X, y)
coef = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title='Feature Importance')
