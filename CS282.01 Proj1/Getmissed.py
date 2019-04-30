import numpy as np
import pandas as pd
from pandas import Series, DataFrame

def GetTrain(train_df):
    all_4=pd.concat([train_df['profession'],train_df['marital']],axis=1)
    all_4=pd.concat([all_4,train_df['schooling']],axis=1)
    all_4=pd.concat([train_df['custAge'],all_4],axis=1)
    
    all_4notnull=all_4.loc[(all_4.custAge.notnull())]
    all_4notnull = all_4notnull.loc[(all_4notnull.schooling.notnull())]
    
    return all_4notnull
    
def GetAgemissed(train_df):
    all_4=pd.concat([train_df['profession'],train_df['marital']],axis=1)
    all_4=pd.concat([all_4,train_df['schooling']],axis=1)
    all_4=pd.concat([train_df['custAge'],all_4],axis=1)

    nullage = all_4.loc[(all_4.custAge.isnull())]
    nullage = nullage.loc[(nullage.schooling.notnull())]

    return nullage

def GetSchoolingmissed(train_df):
    all_4=pd.concat([train_df['profession'],train_df['marital']],axis=1)
    all_4=pd.concat([all_4,train_df['schooling']],axis=1)
    all_4=pd.concat([train_df['custAge'],all_4],axis=1)

    nullschooling = all_4.loc[(all_4.schooling.isnull())]
    nullschooling = nullschooling.loc[(nullschooling.custAge.notnull())]

    return nullschooling

def GetAgeSchoolingmissed(train_df):
    all_4=pd.concat([train_df['profession'],train_df['marital']],axis=1)
    all_4=pd.concat([all_4,train_df['schooling']],axis=1)
    all_4=pd.concat([train_df['custAge'],all_4],axis=1)

    nullageschooling = all_4.loc[(all_4.schooling.isnull())]
    nullageschooling = nullageschooling.loc[(nullageschooling.custAge.isnull())]

    return nullageschooling

