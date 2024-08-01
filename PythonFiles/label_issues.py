from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from cleanlab.filter import find_label_issues
import simple_characteristics
import pandas as pd

def find_labels(dataset):
    class_label = simple_characteristics.get_labels(dataset)
    cat_features = dataset.select_dtypes("category").columns
    num_features = dataset.select_dtypes("float64").columns
    if(len(cat_features) != 0 or len(num_features)) :

        X_encoded = pd.get_dummies(dataset, columns=cat_features)


        scaler = StandardScaler()
        X_scaled = X_encoded.copy()

        X_scaled[num_features] = scaler.fit_transform(X_encoded[num_features])
    
        clf1 = LogisticRegression(solver='lbfgs', max_iter=1000)
        clf2 = svm.SVC(probability=True) # Linear Kernel
        clf3 = KNeighborsClassifier(n_neighbors = class_label.nunique())

        num_crossval_folds = 5  
        pred_probs1 = cross_val_predict(clf1, X_scaled, class_label, cv=num_crossval_folds, method="predict_proba")
        pred_probs2 = cross_val_predict(clf2, X_scaled, class_label, cv=num_crossval_folds, method="predict_proba")
        pred_probs3 = cross_val_predict(clf3, X_scaled, class_label, cv=num_crossval_folds, method="predict_proba" )
        print(len(pred_probs1), pred_probs2)
        #ranked_label_issues1 = find_label_issues(labels=class_label, pred_probs=pred_probs1, return_indices_ranked_by="self_confidence"  )
        #ranked_label_issues2 = find_label_issues(labels=class_label, pred_probs=pred_probs2, return_indices_ranked_by="self_confidence"  )
        #ranked_label_issues3 = find_label_issues( labels=class_label, pred_probs=pred_probs3, return_indices_ranked_by="self_confidence"    )

       # return list(set(ranked_label_issues1).intersection(set(ranked_label_issues2)).intersection(set(ranked_label_issues3)))
    else:
        return 0
