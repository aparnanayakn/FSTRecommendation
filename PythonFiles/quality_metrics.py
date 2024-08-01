import imblearn
import pandas as pd
import simple_characteristics
import pre_processing
import statistics
from collections import Counter
import re
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from cleanlab.filter import find_label_issues


#Classs imbalance

def calculate_class_imbalance(dataset):
    X,y = _split_labelcolumn(dataset)
    class_label = simple_characteristics.get_labels(dataset)

    nUnique = simple_characteristics.count_unique_labels(dataset)
    y = class_label
    class_counts = y.value_counts()
    majority_class_count = class_counts.max()
    imbalance_ratios = {}
    for cls, count in class_counts.items():
        imbalance_ratio = count / majority_class_count
        imbalance_ratios[cls] = imbalance_ratio
    imbalance_classes = []
    for k,i in imbalance_ratios.items():
        if(i<0.6):
            imbalance_classes.append(k)
    return imbalance_classes, len(imbalance_classes) / nUnique


def _split_labelcolumn(dataset):
    class_label = simple_characteristics.get_labels(dataset)
    return dataset.drop(index = simple_characteristics.class_index), dataset.iloc[:, simple_characteristics.class_index]


#Completeness
def completeness(dataset):
    total_missing = dataset.isnull().sum().sum()
    missing_columns_head = dataset.columns[dataset.isna().any()].tolist()
    missing_column_index = []    
    for i in missing_columns_head:
      missing_column_index.append(dataset.columns.get_loc(i))
    if(total_missing == 0):
        missing_cells = []
        for row_index, row in dataset.iterrows():
            for col_index, value in row.items():
                if value == '' or value == ' ?' or value == '?':
                    total_missing += 1
                    missing_cells.append((row_index, col_index))
           
        missing_column_index = list(set(col_index for _, col_index in missing_cells))

    total_cells = len(dataset.axes[0]) * len(dataset.axes[1])
    missing_percentage = total_missing / total_cells

    return missing_percentage, missing_column_index

#Conciseness
def conciseness(dataset):
    uniques = dataset.drop_duplicates(keep='first')
    duplicate = dataset[dataset.duplicated()]
    return (1 - (uniques.shape[0] * uniques.shape[1]) /(dataset.shape[0] * dataset.shape[1]))


#Accuracy
def type_check(singleCol):
    ci=cs=co=cf=cd=cu=cn=0
    intType = re.compile(r"^\d+$")
    dateType1 = re.compile(r"[0-9]{4}[-/][0-9]?[0-9]?[-/][0-9]?[0-9]?")
    dateType2 = re.compile(r"[0-9]?[0-9]?[-/][0-9]?[0-9]?[-/][0-9]{4}")
    stringType = re.compile("^[a-zA-Z]+.*\s*[a-zA-Z]*$")
    floatType = re.compile(r"[+-]?([0-9]*[.])?[0-9]+")
    uriType = re.compile(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))")

    for i in range(len(singleCol)):
        if((uriType.match(str(singleCol[i])))):
            cu+=1
        elif(stringType.match(str(singleCol[i])) and str(singleCol[i]) != "nan"):
            cs+=1
        elif((intType.match(str(singleCol[i])))):
            ci+=1
        elif(dateType1.match(str(singleCol[i]) or dateType2.match(str(singleCol[i])))):
            cd+=1
        elif(floatType.match(str(singleCol[i]))):
            cf+=1
        elif((stringType.match(str(singleCol[i])) and str(singleCol[i]) == "nan")):
            cn+=1
        else:
            co+=1
    daConsidered=['int','str','float','date','uri','other']

    if(cf > ci):             #column with float values, int gets assigned to ci, coverting it to cf
        cf = cf+ci
        ci=0
    
    #return overall.index(max(overall))
    overall=[ci,cs,cf,cd,cu,co]
    total_len = max(overall)
    if(max(overall) < len(singleCol) and cn!=0):
        total_len += cn
    return total_len


def syntax_accuracy(dataset):
    count = 0
    invalid = 0
    invalid_columns = []
    for i in range((dataset.shape[1])):
        flag=0
        count = type_check(dataset.iloc[:, i])
        if(count != dataset.shape[0]):
            invalid+=1
            invalid_columns.append(i)
            
    return (invalid/dataset.shape[1]),invalid_columns

def find_labelissues(dataset):
    class_label = simple_characteristics.get_labels(dataset)
  #  print(type(class_label))
    le = LabelEncoder()
    class_label = pd.Series(le.fit_transform(class_label))
   # print((class_label))
    #X, y = _split_dataset(dataset)
    cat_features = dataset.select_dtypes("category").columns
    num_features = dataset.select_dtypes("float64").columns
    #print("Y",dataset.shape)
    #if(simple_characteristics.class_index in cat_features):
     #   print("CAT",type(cat_features))
    if(len(cat_features) != 0 or len(num_features)) :

        X_encoded = pd.get_dummies(dataset, columns=cat_features)

        scaler = StandardScaler()
        X_scaled = X_encoded.copy()
      #  print(X_scaled.shape)
        X_scaled[num_features] = scaler.fit_transform(X_encoded[num_features])
    
        clf1 = LogisticRegression()
        clf2 = svm.SVC(probability=True) # Linear Kernel
        clf3 = KNeighborsClassifier(n_neighbors = len(np.unique(class_label)))
    
        num_crossval_folds = 5  
        pred_probs1 = cross_val_predict(clf1, X_scaled, class_label, cv=num_crossval_folds, method="predict_proba")
        pred_probs2 = cross_val_predict(clf2, X_scaled, class_label, cv=num_crossval_folds, method="predict_proba")
        pred_probs3 = cross_val_predict(clf3, X_scaled, class_label, cv=num_crossval_folds, method="predict_proba" )
        ranked_label_issues1 = find_label_issues(labels=class_label, pred_probs=pred_probs1, return_indices_ranked_by="self_confidence"  )
        ranked_label_issues2 = find_label_issues(labels=class_label, pred_probs=pred_probs2, return_indices_ranked_by="self_confidence"  )
        ranked_label_issues3 = find_label_issues( labels=class_label, pred_probs=pred_probs3, return_indices_ranked_by="self_confidence"    )
#        return 1
        return list(set(ranked_label_issues1).intersection(set(ranked_label_issues2)).intersection(set(ranked_label_issues3)))
    else:
        return []


def compute_class_overlap(dataset):
    class_overlap_points = []
    class_overlap_percentage = 0
    outlier_points = []
    outliers = 0
    
    km = KMeans(n_clusters=simple_characteristics.count_unique_labels(dataset))
    clusters = km.fit_predict(dataset)
    
    centroids = km.cluster_centers_
    
    distances = cdist(dataset, centroids, 'euclidean')
    min_distances = np.min(distances, axis=1)
    
    cluster_distances = pd.DataFrame({'Cluster': clusters, 'Distance': min_distances})
    
    cluster_statistics = cluster_distances.groupby('Cluster')['Distance'].agg([np.mean, np.std])
    
    for i in range(len(dataset)):
        cluster = clusters[i]
        distance = min_distances[i]
        cluster_mean = cluster_statistics.loc[cluster, 'mean']
        cluster_std = cluster_statistics.loc[cluster, 'std']
        
        if distance > (cluster_mean + 3 * cluster_std):
            outliers += 1
            outlier_points.append(i)
            for other_cluster in range(len(centroids)):
                if other_cluster != cluster:
                    other_cluster_center = centroids[other_cluster]
                    other_cluster_mean = cluster_statistics.loc[other_cluster, 'mean']
                    other_cluster_std = cluster_statistics.loc[other_cluster, 'std']
                    other_cluster_distance = cdist([dataset.iloc[i]], [other_cluster_center], 'euclidean')
                    if other_cluster_distance <= (other_cluster_mean + 3 * other_cluster_std):
                        class_overlap_points.append(i)
                        break
                        
    total_elements = dataset.shape[0]
    total_features = dataset.shape[1]
    class_overlap_percentage = len(class_overlap_points) / (total_elements * total_features)
    
    return class_overlap_points, class_overlap_percentage, outlier_points, outliers/total_elements

