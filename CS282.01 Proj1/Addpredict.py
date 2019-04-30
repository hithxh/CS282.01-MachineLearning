import numpy as np
import pandas as pd
from pandas import Series, DataFrame

def Add_schooling(train_df,data,index):
    for i in range(0,len(index)):
        train_df.loc[index[i],['schooling']]=data[i]
    return train_df

def Add_age(train_df,data,index):
    for i in range(0,len(index)):
        train_df.loc[index[i],['custAge']]=data[i]
    return train_df

