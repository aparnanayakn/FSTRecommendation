import simple_characteristics
import math
import pre_processing
import numpy as np
from statistics import mean, stdev
from sklearn.metrics import mutual_info_score
np.seterr(divide='ignore', invalid='ignore')
from math import log, e

def compute_class_entropy(dataset):
   # dataset = custom_csv(filePath)
    class_label = simple_characteristics.get_labels(dataset)
    entropy=0
    rows = simple_characteristics.read_rows(dataset)
    uc = simple_characteristics.count_unique_labels(dataset)
    values, counts = np.unique(class_label, return_counts=True)
    for i in range(len(values)):
        p = counts[i] / rows
        entropy -= p * math.log(p,uc)
    return entropy

def _mutual_info(x,y,bins):
    c_xy = np.histogram2d(x,y,bins)[0]
    return mutual_info_score(None,None, contingency=c_xy)

def mutual_info(dataset):
    mi = []
    class_index = simple_characteristics.class_index
    categorical_data = pre_processing.category_index(dataset)
    for i in (categorical_data):
        if(i!=simple_characteristics.class_index):
            mi.append(_mutual_info(dataset.iloc[:,i],dataset.iloc[:,class_index],2))
    return mi

def snr(dataset):
    noise_ratio = 0
    class_label = simple_characteristics.get_labels(dataset)
    mi = mutual_info(dataset)
    attr_entropy = compute_attribute_entropy(dataset)
    try:
        if(mean(mi)):
            noise_ratio = (mean(attr_entropy)-mean(mi))/mean(mi)
    except:
        noise_ratio = mean(attr_entropy)
    return noise_ratio

def signaltonoise(a, axis=0, ddof=0.0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    try:
        sd = a.std(axis=axis, ddof=ddof)
    except TypeError as e:
        print(f"Error calculating standard deviation: {e}")
        return None, None, None
    temp = np.where(sd == 0, 0, m/sd)
    return np.std(temp), np.mean(temp), temp


def class_entropy(dataset, base=2):
    class_label = simple_characteristics.get_labels(dataset)
    n_labels = len(class_label)
    if n_labels <= 1:
        return 0
    value,counts = np.unique(class_label, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.
  # Compute entropy
    base = e if base is None else base
    entropy = -np.sum(probs * np.log(probs) / np.log(base))
    return entropy


def _compute_attribute_entropy(dataset,i,c):
    attribute = dataset.iloc[:,i]
    entropy=0
    ig = 0
    rows = simple_characteristics.read_rows(dataset)
    ent=0
    prob = 0
    temp_entropy = 0
    l = -1
    values, counts = np.unique(attribute, return_counts=True)
    for j in (values):
        l+=1
        a = rows-sum(dataset.iloc[:,i] == j)
        if( a == 0 ):
            ent += 0
        else:
            new_df=(dataset.loc[(dataset.iloc[:,i] == j)])
            values1, counts1 = np.unique(new_df.iloc[:,c], return_counts=True)
            for k in range(len(values1)):
                prob += (counts1[k]/sum(counts1)) + math.log(counts1[k]/sum(counts1),2)
            temp_entropy+=(prob*(counts[l]/rows))
            prob = 0
    return temp_entropy

def compute_attribute_entropy(dataset):
    attr_entropy = 0
    cEntropy = class_entropy(dataset)
    class_index = simple_characteristics.class_index
    class_attr = dataset.iloc[:,class_index]
    category_idx = pre_processing.category_index(dataset)
    for i in category_idx:
        attribute = dataset.iloc[:,i]
        YX = np.c_[class_attr, attribute]
       # attr_entropy += cEntropy -        
# attr_entropy += (cEntropy - _compute_attribute_entropy(dataset, i, class_index))
        #attr_entropy.append(cEntropy - _compute_attribute_entropy(dataset, i, class_index))
    
    return attr_entropy/len(category_idx)

def entropy(Y, base=2):
    value,counts = np.unique(Y, return_counts=True)
    probs = counts / len(Y)
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    ent = 0.
  # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)
    return ent


def j_entropy(dataset, class_index, index):
    attribute = dataset.iloc[:,index]
    class_attr = dataset.iloc[:,class_index]
    YX = np.c_[class_attr, attribute]
    return entropy(YX)


def mean_mutual_info(dataset):
    attr_entropy = 0
    cEntropy = class_entropy(dataset)
    class_index = simple_characteristics.class_index
    category_idx = pre_processing.category_index(dataset)
    for i in category_idx:
        attr_entropy+=(cEntropy - j_entropy(dataset, class_index, i))
    if len(category_idx):    
        avg_attr_entropy = attr_entropy / len(category_idx)
        return avg_attr_entropy
    else:
        return 0

def ena_attributes(dataset): #new
    cEntropy = class_entropy(dataset)
    attr_ent = mean_mutual_info(dataset)
    if(attr_ent!=0):
      return cEntropy/attr_ent
    return 0

def ena_attributes2(dataset): #old
    cEntropy = class_entropy(dataset)
    attr_ent = compute_attribute_entropy(dataset)
    if((attr_ent)!=0):
      return cEntropy/attr_ent
    return 0


