# import the libraries 
import pandas as pd
import quandl 
import math

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

#LABELS 
#some point in the future, the price

forecast_col= 'Close'
df.fillna(-99999, inplace=True) #in pandas, na is not #  will be treated as an outliear because in ML you don't want let go of data 

forecast_out= int(math.ceil(0.1*len(df))) #math.ceil will take anything and get to the ceiling , rond everything up to nearest whole, predicting out 10% of the df
 
df['label']= df[forecast_col].shift(-forecast_out) #shifting the columns negatively
#the label column for each row would be  adjusted close price 10 days into the future 

df.dropna(inplace=True)
print(df.tail())