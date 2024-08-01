import pandas as pd
import jenkspy
import numpy as np

dict_values = {}

#Metrics:
#1. Completeness - == 0 if complete ; else incomplete
#2. Conciseness  - == 0 if concise ; else not concise
#3. Syntax accuracy - ==0 if accurate ; else not accurate
#4. Class imbalance - ==0 if balanced ; else imbalance
#5. Labelissues - ==0 if no label issues ;  else labe issues 
#6. Class overlap - == 0 if no overlap ; else overlap
#7. Outliers - == 0 if no outliers ; else outliers


def encode_values(column_values):
    return np.where(column_values > 0, 'Yes','No')

g1 = ['Completeness', 'Conciseness']
#SyntaxAccuracy deleted

g2 = [ 'cor.mean', 'cor.sd', 'cov.mean', 'cov.sd', 'eigenvalues.mean', 'eigenvalues.sd',
       'g_mean.mean', 'g_mean.sd', 'h_mean.mean', 'h_mean.sd', 'iq_range.mean',
       'iq_range.sd', 'kurtosis.mean', 'kurtosis.sd', 'mad.mean', 'mad.sd',
       'max.mean', 'max.sd', 'mean.mean', 'mean.sd', 'median.mean',
       'median.sd', 'min.mean', 'min.sd', 'nr_cor_attr', 'nr_norm',
       'nr_outliers', 'range.mean', 'range.sd', 'sd.mean', 'sd.sd',
       'skewness.mean', 'skewness.sd', 'sparsity.mean', 'sparsity.sd',
       't_mean.mean', 't_mean.sd', 'var.mean', 'var.sd', 'snr.mean','snr.sd' ]

g3 = ['ClassImbRatio', 'OutlierPerc', 'ClassOverlapPerc','LabelIssues']

g4 = ['attr_to_inst',  'inst_to_attr', 'nr_attr', 'nr_inst', 'nr_num', 'nr_bin', 'nUnique']

g5 = ['attr_conc.mean', 'attr_conc.sd', 'attr_ent.mean','attr_ent.sd', 'cEntropy','ena']

def generate_categories(df):
    modified_df = pd.DataFrame()
    
    #g1
    for item in g1:
        modified_df[item+'_bins'] = encode_values(df[item]) #complete if 0 #concise if 0 #accurate if 0
    labels = ['small', 'medium', 'big']
    twoLabels = ['small','big']
    #g2
    for item in g2:
        df[item] = df[item].fillna(0)
        break_item = jenkspy.jenks_breaks(df[item], n_classes = 2)
        modified_df[item+'_bins'] = pd.cut(df[item], bins=break_item, labels=twoLabels, include_lowest=True)
        dict_values[item]=break_item
    #g3
    temp = []
    for i in df['LabelIssues']:
      if i == "[]":
        temp.append("No")
      else:
        temp.append("Yes")
    
    modified_df['LabelIssues_bins'] = temp

    for item in g3[0:len(g3)-1]:
        modified_df[item+"_bins"] = encode_values(df[item]) #class_overlap if 0 
#outliers if 0 #imbalance if 0

    for item in g4[0:len(g4)]:
        break_item = jenkspy.jenks_breaks(df[item], n_classes = 2)
        modified_df[item+'_bins'] = pd.cut(df[item], bins=break_item, labels=twoLabels, include_lowest=True)
        dict_values[item]=break_item

    for item in g5[0:len(g5)-1]:
        break_item = jenkspy.jenks_breaks(df[item], n_classes = 2)
        modified_df[item+'_bins'] = pd.cut(df[item], bins=break_item, labels=twoLabels, include_lowest=True)
        dict_values[item]=break_item

    break_item = jenkspy.jenks_breaks(df['ena'], n_classes = 3)
    modified_df['ena_bins'] = pd.cut(df['ena'], bins=break_item, labels=twoLabels, duplicates='drop')
    dict_values['ena'] = break_item

    modified_df['ena_bins'] = modified_df['ena_bins'].fillna(str('small'))

    modified_df['File'] = df['File']

    dict_values['file'] = df['File']

    return modified_df, dict_values
