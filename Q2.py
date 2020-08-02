import pandas as pd
import numpy as np
from sklearn import linear_model



data = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected_2.csv',thousands = ',')

regression = linear_model.LinearRegression()
High_temp = data['High Temp (°F)']
Low_temp = data['Low Temp (°F)']
pre = data['Precipitation']
total = data['Total']


regression.fit(data[['High Temp (°F)','Low Temp (°F)','Precipitation']], total)
intercept =regression.intercept_
coeff = regression.coef_
print(intercept)
print(coeff)





