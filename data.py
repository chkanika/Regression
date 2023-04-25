# import the libraries 
import pandas as pd
import quandl 

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



# #now define the only columns that we care about 
df = df[['Close','HL_PCT','PCT_change']]

print(df.head())

#can make more features
#got our features 
#features are attrubutes that make up the label, and the label is some sort of prediction, so will the close be a label or a prediction? 
