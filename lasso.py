#2020MCB1237
#Harpreet Singh
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

data = pd.read_csv(r'Hitters.csv').dropna()

columns = list(data)
columns[len(data.columns)-1],columns[len(data.columns)-2] = columns[len(data.columns)-2],columns[len(data.columns)-1] #assigning last column as salary column to simplify calculations
data = data[columns]

X = data.iloc[:,1:len(data.columns)-1].values #taking data from 1-index to 20th index as input
y = data.iloc[:,[len(data.columns)-1]].values #taking data from last index as expected result
std_sal = np.std(y)
mean_sal = np.mean(y)

print(std_sal,mean_sal)
#ColumnTransformation 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

c1 = 13 #these are qualitative columns needed to be transformed 
c2 = 14
c3 = 18

ct = ColumnTransformer([('encoder',OneHotEncoder(),[c1,c2,c3])],remainder = 'passthrough') #Refining and preprocessing of data given in string format
X = np.array(ct.fit_transform(X))

from sklearn.preprocessing import StandardScaler, Normalizer #normalizing the data
scalar = StandardScaler()
X = scalar.fit_transform(X)
X = Normalizer().fit_transform(X)
sc = StandardScaler()
y = sc.fit_transform(y)


def lassoRegression(X,y,testSize):
    ##Lasso Regression
    #lambda = 1.1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize,random_state = 0)

    results=[]
    maxx_score = 0             
    best_alpha = 0
    alphas=[]
    for i in range(1,1301):
        alph = i/1000  #range for alpha (0.01,10)
        alphas.append(alph)
        reg = Lasso (alpha = alph)
        reg.fit(X_train,y_train)
        scr = reg.score(X_test,y_test)
        results.append(scr)
        if maxx_score < scr:
            maxx_score = scr
            best_alpha = alph
    
    print(f"This is optimal lambda for lasso regression : {best_alpha} for getting best score of {maxx_score}")
    reg = Lasso(alpha = best_alpha)
    reg.fit(X_train,y_train)
    coeff = reg.coef_
    print(coeff)
    y_pred = reg.predict(X_test)    
    for i in range(0,len(y_test)):
        y_test[i] = std_sal*y_test[i] + mean_sal

    for i in range(0,len(y_pred)):
        y_pred[i] = std_sal*y_pred[i] + mean_sal

    print(f"These are the predicted salaries")
    print(y_pred)

    print("These are the actual salaries")
    print(y_test)

    plt.title("Alpha value-R^2 - Lasso Regression")
    plt.xlabel("Alpha values")
    plt.ylabel("R^2 values")
    plt.plot(alphas,results)

    plt.figure(figsize = (12,10)) #checking for correlation of columns as heatmap 
    sns.heatmap(data[columns[:-1]].corr())
    plt.show()


def ridgeRegression(X,y,testSize): 
    #Ridge regression 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize,random_state = 0)

    results=[]
    maxx_score = 0             
    best_alpha = 0
    alphas = []
    for i in range(1,1201): #checking the optimal alpha for getting best results 
        alph = i/1000
        alphas.append(i/1000)
        reg = Ridge(alpha = alph)
        reg.fit(X_train,y_train)
        scr = reg.score(X_test,y_test)
        results.append(scr)
        if maxx_score < scr:
            maxx_score = scr
            best_alpha = alph
    
    print(f"This is optimal lambda for ridge regression : {best_alpha} for getting best score of {maxx_score}")
    reg = Ridge(alpha = best_alpha)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    
    print(f"These are the predicted and actual salaries")

    for i in range(0,len(y_test)):
        y_test[i] = std_sal*y_test[i] + mean_sal

    for i in range(0,len(y_pred)):
        y_pred[i] = std_sal*y_pred[i] + mean_sal

    print(f"These are the predicted salaries")
    print(y_pred)

    print("These are the actual salaries")
    print(y_test)

    coeff = reg.coef_
    print(coeff)
    plt.title("Alpha value-R^2 - Ridge Regression")
    plt.xlabel("Alpha values")
    plt.ylabel("R^2 values")
    plt.plot(alphas,results)
    plt.show()
    

    



ridgeRegression(X,y,0.2)#ridge regresion 
lassoRegression(X,y,0.2)#lasso regression


print(data.describe())#data summary 