#用svm对上证指数进行预测
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import cross_validation
import matplotlib.pyplot as plt

#import sys
#reload(sys)
#sys.setdefaultencoding('gbk')

data=pd.read_csv('stock.csv',encoding='gbk',parse_dates=[0],index_col=0)
data.sort_index(0,ascending=True,inplace=True)

tdata=data[0:-2]
print(tdata.head())
dayfeature=150#用过去 150天的数据作为特征
featurenum=5*dayfeature#特征个数
x=np.zeros((tdata.shape[0]-dayfeature,featurenum+1))
y=np.zeros((tdata.shape[0]-dayfeature))

for i in range(0,tdata.shape[0]-dayfeature):
        x[i,0:featurenum]=np.array(tdata[i:i+dayfeature][[u'收盘价',u'最高价',u'最低价',u'开盘价',u'成交量']]).reshape((1,featurenum))
        x[i,featurenum]=data.ix[i+dayfeature][u'开盘价']
        

for i in range(0,tdata.shape[0]-dayfeature):
        
        if data.ix[i+dayfeature][u'收盘价']>=data.ix[i+dayfeature-1][u'收盘价']:
                y=1
        else:
                y=0
                

clf=svm.SVC(kernel='rbf')
clf.fit(x,y)
svc_scores=cross_validation.cross_val_score(clf,x,y,cv=5)
print("svm classifier accuacy:")
print (svc_scores)
re=clf.predict(x)
print (re)

test=np.zeros((2,featurenum+1))
test[0,0:featurenum]=np.array(data[-dayfeature::][[u'收盘价',u'较高价',u'较低价',u'开盘价',u'成交量']]).reshape((1,featurenum))
test[0,featurenum]=data.ix[-1][u'开盘价']

test[1,0:featurenum]=np.array(data[-dayfeature-1:-1][[u'收盘价',u'较高价',u'较低价',u'开盘价',u'成交量']]).reshape((1,featurenum))
test[1,featurenum]=data.ix[-2][u'开盘价']

predict=clf.predict(test)
print (predict)
