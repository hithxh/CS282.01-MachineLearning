import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge

def Predict_schooling_logistic(train,test,kind):
    n=test.shape[1]
    train=pd.get_dummies(train)
    if n==17:
        for i in range(17,25):
            logistic_model = LogisticRegression()
            logistic_model.fit(train[train.columns[0:17]],train[train.columns[i]])
            fitted_test = logistic_model.predict_proba(test[test.columns[0:17]])[:, 1]
            if i==17:
                r=fitted_test
            else:
                r=np.vstack((r,fitted_test))
        r=r.T
        for j in range(0,r.shape[0]):
            maxp=max(r[j])
            r[j]=r[j]-maxp
            r[j][r[j]>=0]=1
            r[j][r[j]<0]=0
            r_list=r[j].tolist()
            if j==0:
                Pred_schooling=[kind[r_list.index(1)]]
            else:
                Pred_schooling.append(kind[r_list.index(1)])
        Pred_schooling_index=test.index.tolist()
        return Pred_schooling_index,Pred_schooling
    else:
        for i in range(17,25):
            logistic_model = LogisticRegression()
            logistic_model.fit(train[train.columns[1:17]],train[train.columns[i]])
            fitted_test = logistic_model.predict_proba(test)[:, 1]
            if i==17:
                r=fitted_test
            else:
                r=np.vstack((r,fitted_test))
        r=r.T
        for j in range(0,r.shape[0]):
            maxp=max(r[j])
            r[j]=r[j]-maxp
            r[j][r[j]>=0]=1
            r[j][r[j]<0]=0
            r_list=r[j].tolist()
            if j==0:
                Pred_schooling=[kind[r_list.index(1)]]
            else:
                Pred_schooling.append(kind[r_list.index(1)])
        Pred_schooling_index=test.index.tolist()
        return Pred_schooling_index,Pred_schooling
    
def Predict_age_ridge(train,test):
    n=test.shape[1]
    pred_index=test.index
    if n==16:
        y_train=train['custAge']
        X_train=pd.get_dummies(train.drop(columns=['custAge','schooling']))
        clf = Ridge(alpha=1)
        clf.fit(X_train,y_train)
        pred=clf.predict(test)
        return pred_index,pred
    else:
        y_train=train['custAge']
        X_train=pd.get_dummies(train.drop(columns=['custAge']))
        clf = Ridge(alpha=1)
        clf.fit(X_train,y_train)
        pred=clf.predict(test)
        return pred_index,pred

