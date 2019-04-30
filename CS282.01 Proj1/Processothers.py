import numpy as np
import pandas as pd
from pandas import Series, DataFrame

def Process_dayofweek(train_df):
    train_df['day_of_week']=train_df['day_of_week'].fillna(method='pad')
    return train_df

def Process_campaign(train_df):
    value_of_campaign=train_df['campaign']
    value_of_campaign=value_of_campaign.tolist()
    collist_of_train_df=train_df.columns.tolist()
    campaign_index=collist_of_train_df.index('campaign')
    train_df.insert(campaign_index+1,'campaign_days',value_of_campaign)
    notdays_df=train_df[train_df.campaign_days<=10]
    notdays_index=notdays_df.index
    train_df.loc[notdays_index,'campaign_days']=0  
    nottimes_df=train_df[train_df.campaign>10]
    nottimes_index=nottimes_df.index
    train_df.loc[nottimes_index,'campaign']=0
    return train_df

def Process_pdays_pmonths(train_df):
    pdays999_df=train_df[train_df.pdays==999]
    pdays_index=pdays999_df.index
    train_df.loc[pdays_index,'pdays']=0
    pmonths999_df=train_df[train_df.pmonths==999]
    pmonths_index=pmonths999_df.index
    train_df.loc[pmonths_index,'pmonths']=0
    return train_df

def Process_get_class(kind):
    kind=kind.drop_duplicates()
    kind=kind.tolist()
    kind=sorted(kind)
    return kind

