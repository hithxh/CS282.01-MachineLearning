import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression,Lasso,Ridge,BayesianRidge
from sklearn.metrics import r2_score
from sklearn import cross_validation



def judge_process(train_V1):
    pro_features = ['custAge','campaign','campaign_days','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m',
                    'nr.employed','pmonths','pastEmail']
    features = train_V1.columns.values.tolist()
    scaler1 = preprocessing.MinMaxScaler()
    scaler2 = preprocessing.StandardScaler()
    for term in features:
        if term in pro_features:
            if min(train_V1[term].values)>=0:
                train_V1[term] = scaler1.fit_transform(train_V1[[term]])
            else:
                train_V1[term] = scaler2.fit_transform(train_V1[[term]])
                
    return train_V1   

def profit(X_train,Y_profit):
    train_V1 = judge_process(X_train)
    train_V1 = train_V1.loc[(Y_profit != 0)]
    X_train = train_V1.values[:,:]
    Y_train = Y_profit[(Y_profit !=0)]
    lr = LinearRegression()  
    scores1 = cross_validation.cross_val_score(lr, X_train, Y_train, cv=10, scoring = 'r2')
    lasso = Lasso(alpha= 0.1)
    scores2 = cross_validation.cross_val_score(lasso, X_train, Y_train, cv=10, scoring = 'r2')
    rgr = Ridge(alpha=0.5)
    scores3 = cross_validation.cross_val_score(rgr, X_train, Y_train, cv=10, scoring = 'r2')
    br = BayesianRidge()
    scores4 = cross_validation.cross_val_score(br, X_train, Y_train, cv=10, scoring = 'r2')
    print('The accuracy by LinearRegression is', scores1.mean())
    print('The accuracy by Lasso Regression is', scores2.mean())
    print('The accuracy by RidgeRegression is', scores3.mean())
    print('The accuracy by BayesianRegression is', scores4.mean())

def select(df, coef):
    drop_weit = np.where(coef ==0)
    drop_feature = df.columns[drop_weit]
    remain = df.drop(drop_feature,axis=1, inplace=False)
    return remain
    
    

