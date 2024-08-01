import numpy as np
import math
from scipy.stats import norm, kurtosis
from collections import Counter
import simple_characteristics
import pandas as pd


class_index = 0

def compute_skew(df):
    skew_values =  df.skew(axis=0)
    sym = fsym = asym = 0
    for i in skew_values:
        if( i > -.5 and i < .5):
            sym +=1
        elif( i < 1 and i > -1):
            fsym += 1
        else:
            asym +=1
    return  sum(skew_values) / len(skew_values), sym/len(skew_values), asym/len(skew_values), fsym/len(skew_values)

def compute_kurt(df):
    kurt_values = df.kurt(axis=0)
    if (sum(kurt_values)/len(kurt_values) > 3):
        k_type = 'high'
    elif (sum(kurt_values)/len(kurt_values) < 3 and sum(kurt_values)/len(kurt_values) > 0):
        k_type = 'low'
    else:
        k_type = 'medium' 
    return k_type, sum(kurt_values)/len(kurt_values)

def compute_corr(df):
    c1 = df.corr(method = 'pearson')
    c2 = c1.copy()
    c2.values[np.tril_indices_from(c2)] = np.nan
    avg_corr = c2.unstack().mean()
    neu = neg = pos =0
   # c3 =  [x for x in c2.values.tolist() if str(x) != 'nan']
    
    l = [item for sublist in c2.values.tolist() for item in sublist]
    c3 = [x for x in l if ~np.isnan(x)]
    for i in c3:
        if(i > 0):
            pos +=1
        elif(i < 0):
            neg += 1
        else:
            neu +=1
    corr_list = [pos, neg, neu]
    return corr_list.index(max(corr_list)) , avg_corr,  pos/(df.shape[1]* (df.shape[1]-1)), neg/(df.shape[1]* (df.shape[1]-1)), neu/(df.shape[1]* (df.shape[1]-1))  
    
    
def compute_correlation(dataset):
    sp=p=sn=n=0
       #dataset = custom_csv(filePath)
    rows, cols = dataset.shape
    corr_dataset = dataset.corr() #Compute pairwise correlation of columns, excluding NA/null values.
    su = 0
    corr_numpy = corr_dataset.to_numpy()
    avg_corr =  (corr_numpy.sum() - np.diag(corr_numpy).sum())/ 2
    print(np.shape(corr_numpy)[1])
    p=n=0
    for i in range(np.shape(corr_numpy)[1]):
        for j in range(np.shape(corr_numpy)[1]):
            if(i>j):
                su += corr_numpy[i][j]
                if(corr_numpy[i][j] >= 0):
                    p+=1
                else:
                    n+=1
    print("Dataset",dataset.shape)
    print("Perc", p, n, p/ (dataset.shape[1]+(dataset.shape[1]+1)) / 2,  n/(dataset.shape[1]+(dataset.shape[1]+1))/2 )
    print("SU", su, avg_corr)
    return p, n, su


def dataset_balance(dataset):
    class_labels = simple_characteristics.get_labels(dataset)
    label_count = Counter(class_labels)
    max_value = max(label_count.values())
    min_value = min(label_count.values())
    return min_value/max_value


def avg_coefficient_variation(dataset):
    df1 = dataset[1:].mean()
    df2 = dataset[1:].std()
    #print(df2/df1)
    #print(type(df2), df2.shape)
    return df1.div(df2).sum()/dataset.shape[1]
    
