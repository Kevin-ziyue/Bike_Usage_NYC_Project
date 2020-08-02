import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import re


def main(x_title,y_title,b_name):
    #Importing dataset
    diamonds = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')

    #Feature and target matrices
    X = diamonds[x_title]
    y = diamonds[y_title]

    #Process the data
    X = X.to_numpy()
    p1 = re.compile(r'(\d+\.\d+)\s\([A-Z]\)')

    row1 = 0
    col1 = 1
    for row1 in range(len(X)):
        res1 = p1.fullmatch(X[row1][col1])
        if res1 != None:
            X[row1][col1] = float(res1.group(1))
        elif (X[row1][col1] == 'T'):
            X[row1][col1] = 0.0
        else:
            X[row1][col1] = float(X[row1][col1])
        X[row1][col1] = np.float32(X[row1][col1])
    
    row1 = 0
    col1 = 0      
    for row1 in range(len(X)):
        if X[row1][col1] == 'Monday':
            X[row1][col1] = 1.0
        elif X[row1][col1] == 'Tuesday':
            X[row1][col1] = 2.0
        elif X[row1][col1] == 'Wednesday':
            X[row1][col1] = 3.0
        elif X[row1][col1] == 'Thursday':
            X[row1][col1] = 4.0
        elif X[row1][col1] == 'Friday':
            X[row1][col1] = 5.0
        elif X[row1][col1] == 'Saturday':
            X[row1][col1] = 6.0
        elif X[row1][col1] == 'Sunday':
            X[row1][col1] = 7.0
        X[row1][col1] = np.float32(X[row1][col1])
    
    X = X.tolist()
    X = np.array(X)
    
    #Training and testing split, with 25% of the data reserved as the test set
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.25, random_state=101)

    #Normalizing training and testing data
    [X_train, trn_mean, trn_std] = normalize_train(X_train)
    X_test = normalize_test(X_test, trn_mean, trn_std)

    #Define the range of lambda to test
    lmbda = np.logspace(-1.0, 2.0, num=101, base=10.0, endpoint=False)#fill in
    
    MODEL = []
    MSE = []
    for l in lmbda:
        #Train the regression model using a regularization parameter of l
        model = train_model(X_train,y_train,l)

        #Evaluate the MSE on the test set
        mse = error(X_test,y_test,model)

        #Store the model and mse in lists for further processing
        MODEL.append(model)
        MSE.append(mse)

    #Plot the MSE as a function of lmbda
    plt.plot(lmbda,MSE,color='blue',linewidth=1,label='MSE as function of lmbda') #fill in
    plt.title("MSE versus lambda value")
    plt.xlabel("lambda")
    plt.ylabel("MSE")
    plt.show()

    #Find best value of lmbda in terms of MSE
    ind = MSE.index(min(MSE))#fill in
    [lmda_best,MSE_best,model_best] = [lmbda[ind],MSE[ind],MODEL[ind]]

    print('The MSE for model treating ' + b_name + ' as target value is ' + str(MSE_best) + '. The best lambda tested for it is ' + str(lmda_best))
    
    
    return MSE_best
    

#Function that normalizes features in training set to zero mean and unit variance.
#Input: training data X_train
#Output: the normalized version of the feature matrix: X, the mean of each column in
#training set: trn_mean, the std dev of each column in training set: trn_std.
def normalize_train(X_train):

    #fill in   
    mean = np.mean(X_train, axis = 0)
    std = np.std(X_train, axis = 0)
    X = (X_train - mean) / std

    return X, mean, std


#Function that normalizes testing set according to mean and std of training set
#Input: testing data: X_test, mean of each column in training set: trn_mean, standard deviation of each
#column in training set: trn_std
#Output: X, the normalized version of the feature matrix, X_test.
def normalize_test(X_test, trn_mean, trn_std):

    #fill in
    X = (X_test - trn_mean) / trn_std
    
    return X


#Function that trains a ridge regression model on the input dataset with lambda=l.
#Input: Feature matrix X, target variable vector y, regularization parameter l.
#Output: model, a numpy object containing the trained model.
def train_model(X,y,l):

    #fill in
    model = linear_model.Ridge(alpha = l, fit_intercept = True)
    model.fit(X,y)

    return model


#Function that calculates the mean squared error of the model on the input dataset.
#Input: Feature matrix X, target variable vector y, numpy model object
#Output: mse, the mean squared error
def error(X,y,model):

    #Fill in
    y = np.array(y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y,y_pred)

    return mse

if __name__ == '__main__':
    qu = main(['Day','Precipitation','High Temp (°F)','Low Temp (°F)','Brooklyn Bridge','Manhattan Bridge','Williamsburg Bridge'],['Queensboro Bridge'],'Queensboro Bridge')
    wi = main(['Day','Precipitation','High Temp (°F)','Low Temp (°F)','Brooklyn Bridge','Manhattan Bridge','Queensboro Bridge'],['Williamsburg Bridge'],'Williamsburg Bridge')
    ma = main(['Day','Precipitation','High Temp (°F)','Low Temp (°F)','Brooklyn Bridge','Queensboro Bridge','Williamsburg Bridge'],['Manhattan Bridge'],'Manhattan Bridge')
    br = main(['Day','Precipitation','High Temp (°F)','Low Temp (°F)','Queensboro Bridge','Manhattan Bridge','Williamsburg Bridge'],['Brooklyn Bridge'],'Brooklyn Bridge')
    
