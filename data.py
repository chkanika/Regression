# import the libraries 
import pandas as pd
import quandl 

# look for the dataset and find the ticker
df = quandl.get('BATS/EDGA_TGH_PB')

print(df.head())


