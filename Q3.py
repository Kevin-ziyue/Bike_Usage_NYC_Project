import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def main(datapath, degrees):
	df = pd.read_csv("NYC_Bicycle_Counts_2016_Corrected_2.csv")
	rider = data['Total']
	pre = data['Precipitation']

	for i in range(len(pre)):                    
    	if pre[i] == 'T':
        	pre[i] = 1
		elif: pre[i] = '0'
			pre[i] = 0
		else:
			pre[i] = 1

	for k in range(len(rider)):
    	rider[k] = float(rider[k])

	plt.scatter(rider,pre,marker='*'color='b')

	X_train, X_test, y_train, y_test = train_test_split([rider],pre,train_size=0.8)

	logR = LogisticRegression()
	logR.fit(X_train,y_train)

	logR.predict_proba(X_test)
	y_predicted = logR.predict(X_test)
	score = logR.score(X_test,y_test)

