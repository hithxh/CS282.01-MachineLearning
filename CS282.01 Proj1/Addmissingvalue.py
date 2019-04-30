import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import Getmissed

def AddAge_missedval(train,Agemiss):
    train=train.drop(columns=['custAge'])
    Agemiss=Agemiss.drop(columns=['custAge'])
    train=pd.get_dummies(train)
    Agemiss=pd.get_dummies(Agemiss)
    train_col=train.columns.tolist()
    Agemiss_col=Agemiss.columns.tolist()
    for kind in Agemiss_col:
        train_col.remove(kind)
    lostval = train_col
    train_col=train.columns.tolist()
    for losskind in lostval:
        addval=np.zeros(Agemiss.shape[0])
        addindex=train_col.index(losskind)
        Agemiss.insert(addindex,losskind,addval)
    return Agemiss

def AddSchooling_missedval(train,Schoolingmiss):
    train=train.drop(columns=['schooling'])
    Schoolingmiss=Schoolingmiss.drop(columns=['schooling'])
    train=pd.get_dummies(train)
    Schoolingmiss=pd.get_dummies(Schoolingmiss)
    train_col=train.columns.tolist()
    Schoolingmiss_col=Schoolingmiss.columns.tolist()
    for kind in Schoolingmiss_col:
        train_col.remove(kind)
    lostval = train_col
    train_col=train.columns.tolist()
    for losskind in lostval:
        addval=np.zeros(Schoolingmiss.shape[0])
        addindex=train_col.index(losskind)
        Schoolingmiss.insert(addindex,losskind,addval)
    return Schoolingmiss

def AddAgeSchooling_missedval(train,AgeSchoolingmiss):
    train=train.drop(columns=['schooling','custAge'])
    AgeSchoolingmiss=AgeSchoolingmiss.drop(columns=['schooling','custAge'])
    train=pd.get_dummies(train)
    AgeSchoolingmiss=pd.get_dummies(AgeSchoolingmiss)
    train_col=train.columns.tolist()
    AgeSchoolingmiss_col=AgeSchoolingmiss.columns.tolist()
    for kind in AgeSchoolingmiss_col:
        train_col.remove(kind)
    lostval = train_col
    train_col=train.columns.tolist()
    for losskind in lostval:
        addval=np.zeros(AgeSchoolingmiss.shape[0])
        addindex=train_col.index(losskind)
        AgeSchoolingmiss.insert(addindex,losskind,addval)
    return AgeSchoolingmiss
