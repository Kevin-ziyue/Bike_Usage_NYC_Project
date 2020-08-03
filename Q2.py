import pandas as pd
import numpy as np
from sklearn import linear_model
import statsmodels.api as sm
from sklearn import preprocessing

def main():
	data = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected_2.csv',thousands = ',')

	regression = linear_model.LinearRegression()
	High_temp = data['High Temp (°F)']
	Low_temp = data['Low Temp (°F)']
	total = data['Total']

	##regression on the original data
	regression.fit(data[['High Temp (°F)','Low Temp (°F)','Precipitation']], total)
	intercept =regression.intercept_
	coeff = regression.coef_
	print(intercept)
	print(coeff)
	X = sm.add_constant(data[['High Temp (°F)','Low Temp (°F)','Precipitation']])
	model = sm.OLS(total,X).fit()
	print(model.summary())

	## so we try regression on normalized data
	X_normal= preprocessing.scale(data[['High Temp (°F)','Low Temp (°F)' ,'Precipitation']])
	regression.fit(X_normal, total)
	print(regression.intercept_)
	print(regression.coef_)
	X_normal_1 = sm.add_constant(X_normal)
	model = sm.OLS(total,X_normal_1).fit()
	print(model.summary())
	

if __name__ == '__main__':
	main()


