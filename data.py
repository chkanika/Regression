# import the libraries 
import pandas as pd
import quandl, math
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn import cross_validation
from sklearn import preprocessing, svm #scaling data, feature to be between -1 and +1. 
from sklearn.linear_model import LinearRegression

# look for the dataset and find the ticker
#df = quandl.get('BATS/EDGA_TGH_PB')

#print(df.head())

# making dataframe 
df = pd.read_csv("TSLA.csv") 
   
# output the dataframe
#print(df.to_string())

#make a list 
df = df[['Open', 'High', 'Low', 'Close']]

# #percent volatalitiy 
df['HL_PCT']= (df['High'] - df['Close']) / df['Close'] * 100.0

# #percent change in new - old / old
df ['PCT_change']= (df['Close']- df['Open'])/ df['Open'] * 100.0

# #now define the only columns that we care about, FEATURES 
df = df[['Close','HL_PCT','PCT_change']]


#can make more features
#got our features 
#features are attrubutes that make up the label, and the label is some sort of prediction, so will the close be a label or a prediction? 
#none of the above, it could be a label, if just using pct change ~

#LABELS (comparing forecast price to adjusted price)
#some point in the future, the price

forecast_col= 'Close'
df.fillna(-99999, inplace=True) #in pandas, na is not #  will be treated as an outliear because in ML you don't want let go of data 

forecast_out= int(math.ceil(0.1*len(df))) #math.ceil will take anything and get to the ceiling , rond everything up to nearest whole, predicting out 10% of the df
print(forecast_out)
 
df['label']= df[forecast_col].shift(-forecast_out) #shifting the columns negatively
#the label column for each row would be  adjusted close price 10 days into the future 


df.dropna(inplace=True)

#features 
X = np.array(df.drop(['label'], axis=1))
#labels 
y= np.array(df['label'])

print(len(X), len(y))

X= preprocessing.scale(X)
y= np.array(df['label'])


print (len(X), len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
 #30% of the test size  

clf= LinearRegression(n_jobs=10) #to change algo, just switch linear regresion to svm.SVR(kernel= 'polynomial')
clf.fit(X_train, y_train) #fit= train 
accuracy = clf.score(X_test, y_test) #score = test

print(accuracy)
# accuarcy in linear regression is the squared error 

