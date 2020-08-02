import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn import preprocessing

def main():
	data = pd.read_csv("NYC_Bicycle_Counts_2016_Corrected_2.csv",thousands = ',')
	rider = data['Total']
	pre = data['Precipitation']
	pre = pre.tolist()
	for i in range(len(pre)):
		if pre[i] != 0:
			pre[i] = 1
	plt.scatter(rider,pre,marker='*',color='b')
	plt.show()
	

	X_train, X_test, y_train, y_test = train_test_split(data[['Total']],pre,train_size=0.92,random_state= 5)
	
	# Normalize the data attributes for the dataset.
	# normalize the data attributes
	X_train = preprocessing.scale(X_train)
	print(X_train)
	X_test = preprocessing.scale(X_test)
	X_test = X_test.tolist()
	logR = LogisticRegression()
	logR.fit(X_train,y_train)
	X_test
	print(logR.predict_proba(X_test))
	
	y_predicted = logR.predict(X_test)
	y_predicted = y_predicted.tolist()
	for i in range(len(y_predicted)):
		print("Test " + str(i))
		print("Prdiction by model: " +str(y_predicted[i]))
		print("Actual result: " +str(y_test[i]))
	score = logR.score(X_test,y_test)
	print(score)


if __name__ == '__main__':
	main()

