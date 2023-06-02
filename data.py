# import the libraries 
import pandas as pd
import quandl, math, datetime
import time
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn import cross_validation
from sklearn import preprocessing, svm #scaling data, feature to be between -1 and +1. 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style 
import pickle 

style.use('ggplot')

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




#features 
X = np.array(df.drop(['label'], axis=1))
#labels 
# y= np.array(df['label'])

X= preprocessing.scale(X)
X = X[:-forecast_out] #to the point of negative forecastout 
X_lately = X[-forecast_out:]
df.dropna(inplace=True)

y= np.array(df['label'])
print (len(X), len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #30% of the test size  
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test) #score = test

# print(accuracy)
# accuarcy in linear regression is the squared error 

forecast_set = clf.predict(X_lately) #single value or an array of value to predict 
print(forecast_set, accuracy, forecast_out)

df['Date'] = pd.to_datetime(df.index) 
df['Forecast'] = np.nan

last_date = df['Date'].iloc[-1]
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Close'].plot()

df ['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

