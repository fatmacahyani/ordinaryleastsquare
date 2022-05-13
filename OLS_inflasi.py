# load numpy and pandas for data manipulation
# load statsmodels as alias ``sm``import statsmodels.api as sm
import pandas as pd
import numpy as np
import statsmodels.api as sm

# load the faktor inflasi dataset into a pandas data frame
df=pd.read_csv('faktorinflasi.csv', index_col=0)
df.head(100)

x = df[['Kurs', 'Rate']] #response
y = df['Inflasi'] #predictor

# Add a constant term so that you fit the intercept of your linear model
x = sm.add_constant(x)
x.head(100)

# The statsmodels object has a method called fit() that takes the independent(X ) and dependent(y) values as arguments
# The summary() method is used to obtain a table which gives an extensive description about the regression results
model = sm.OLS(y, x).fit()
summary = model.summary()
print(summary)