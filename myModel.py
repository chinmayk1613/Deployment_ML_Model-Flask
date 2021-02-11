import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Fitting Simple linear Regression to the train set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

def trainLinear():
    #import dataset
    dataset=pd.read_csv('Salary_Data.csv') #load data set
    X=dataset.iloc[:,:-1].values    #independnt variables
    y=dataset.iloc[:, 1].values     #dependent data

    #Splitting dataset into train and test data
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

    regressor.fit(X_train,y_train)  #ft is method name which fit the regreesor to data
    pickle.dump(regressor, open('myModel.pkl', 'wb'))
def prediction():
    model = pickle.load(open('myModel.pkl','rb'))
    print(model.predict([[2.5]]))

if __name__=='__main__':
    #trainLinear()
    prediction()
