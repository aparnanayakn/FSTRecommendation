import numpy as np
import math

class_index = 0

def get_XY(dataset):
    temp = get_labels(dataset)
    return dataset.drop(columns=dataset.columns[class_index]), dataset.iloc[:, class_index]

def get_labels(dataset):
    global class_index
    try:
        flag = 0
        n = dataset.iloc[:, -1].nunique(dropna=False)
        perc = dataset.iloc[:, -1].value_counts(normalize=True)*100
        perc1 = dataset.iloc[:, 0].value_counts(normalize=True)*100
        if(len(perc) > len(perc1)):  # checking whether 1st column is label #change here, > A ;  < AA
            flag = 1
        if(len(perc) > 10 or len(perc) == 1):
            if((len(perc1) > 10 or len(perc1) == 1)): 
                for i in range(dataset.shape[1]):
                    n = dataset.iloc[:, i].nunique(dropna=False)
                    if(n < 10):
                        class_index = i
                        return dataset.iloc[:, i]
            ###Need to add another condition what if none turns out be categorical
        if(flag == 1):
            class_index = 0
            return dataset.iloc[:, 0]
        else:
            class_index = read_columns(dataset)-1
            return dataset.iloc[:, -1]
    except Exception as e:
        print("Can not read last column items for", dataset)

def count_unique_labels(dataset):
    global unique_labels
    try:
        n = get_labels(dataset)
        unique_labels = n.nunique(dropna=False)
        return n.nunique(dropna=False)
    except:
        print("Can not read unique items for", dataset)

def read_rows(dataset):
    try:
    # dataset = custom_csv(filePath)
        return (dataset.shape[0])
    except:
        print("Can not read rows for", dataset)

def read_columns(dataset):
    try:
        return (dataset.shape[1])
    except:
        print("Can not read columns for", dataset)
        
