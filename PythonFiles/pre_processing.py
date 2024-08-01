from sklearn.preprocessing import LabelEncoder
import numpy as np
import math

def _assumption1_categorical(df):
    likely_cat = []
    for idx, var in enumerate(df.columns):
       # print(df[var].nunique(), df[var].count())
        if(1.*df[var].nunique()/df[var].count() < 0.05):  # or some other threshold
            likely_cat.append(idx)
    return likely_cat

def _assumption2_categorical(df):
    top_n = math.floor(df.shape[0]/70) #0.7
    likely_cat = []
    for idx, var in enumerate(df.columns):
        if(1.*df[var].value_counts(normalize=True).head(top_n).sum() > 0.7):  # or some other threshold
            likely_cat.append(idx)
    return likely_cat


def category_index(df):
    le = LabelEncoder()
    ass1 = _assumption1_categorical(df)
    ass2 = _assumption2_categorical(df)

    #extract only columns that belong to
    commonidx = (list(set(ass1) | set(ass2)))
   # for i in commonidx:
    #   df.iloc[:,i] = le.fit_transform(df.iloc[:,i])

    return commonidx


def convert_str_int_nominal(df):
    le = LabelEncoder()
    df = convert_NAs(df)
    for i in range(df.shape[1]):
        if df.iloc[:, i].dtype == object or df.iloc[:, i].dtype == str:
            df.iloc[:, i] = df.iloc[:, i].str.replace("'", "")  # Remove single quotes
            df.iloc[:, i] = le.fit_transform(df.iloc[:, i])
    return df


def convert_str_int_categorical(df):
    le = LabelEncoder()
    ass1 = _assumption1_categorical(df)
    ass2 = _assumption2_categorical(df)
    #extract only columns that belong to
    commonidx = (list(set(ass1) and set(ass2)))
    for i in commonidx:
        if(df.iloc[:, i].dtype == "object"):
            df.iloc[:, i] = df.apply(lambda i: le.fit_transform(i.astype(str)), axis=0, result_type='expand')
    return df

def drop_empty_columns(df):
    df.fillna(value='NULL', inplace=True)
    df = df.dropna(how="all", axis=1, thresh=0.9)
    #print(df.head(5))
    return df


def drop_rows(df):
    perc = 75.0  # Here N is 75
    min_count = int(((100-perc)/100)*df.shape[1] + 1)
    mod_df = df.dropna(axis=0, thresh=min_count)
    return mod_df


def drop_columns(df):
    perc = 75.0  # Here N is 75
    min_count = int(((100-perc)/100)*df.shape[0] + 1)
    mod_df = df.dropna(axis=1, thresh=min_count)
    return mod_df


def convert_NAs(df):
    mod_df=df.replace('?',0)
    mod_df = df.replace(np.nan, 0) 
    return mod_df
