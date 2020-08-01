import pandas as pd
from sklearn import linear_model

data = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected_2.csv')
regression = linear_model.LinearRegression
High_temp = data['High Temp (°F)']
Low_temp = data['Low Temp (°F)']
pre = data['Precipitation']

for i in range(len(pre)):                    
    if pre[i] == 'T':
        pre[i] = '0'
    if pre[i][-3:] == "(S)":
        pre[i] = pre[i][:-4]


regression.fit([High_temp,Low_temp,pre],data.Total)
intercept =reg.intercept_)
coeff = reg.coef_





