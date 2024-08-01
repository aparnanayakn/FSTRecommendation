from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import simple_characteristics

from scipy.spatial.distance import cdist


def compute_class_overlap(dataset):
    m = 0
    s = 0 
    count = 0
    outlier = 0
    flag = 0
    class_overlap = []
   # dataset = custom_csv(filePath)
    km = KMeans(n_clusters = simple_characteristics.count_unique_labels(dataset))
    clusters = km.fit_predict(dataset)
    # points array will be used to reach the index easy
    points = np.empty((0,len(dataset.axes[1])), float)
    # distances will be used to calculseetate outliers
    distances = np.empty((0,len(dataset.axes[0])), float)   
        # getting points and distances
    centroids = km.cluster_centers_
    for i, center_elem in enumerate(centroids):
            # cdist is used to calculate the distance between center and other points
        distances = np.append(distances, cdist([center_elem],dataset[clusters == i], 'euclidean')) 
        points = np.append(points, dataset[clusters == i], axis=0)
        
    cluster_distance_d = {'cluster':clusters, 'distance':distances}
    cluster_distance = pd.DataFrame(cluster_distance_d)

    grouped = cluster_distance.groupby(['cluster'], as_index = False)
    cluster_statistics = grouped[['distance']].agg([np.mean, np.std]) 
    
    for i in range(len(cluster_distance)):#
        for j in range(len(cluster_statistics)):
            if(cluster_statistics.index[j]==cluster_distance.iloc[i,0]):
                m = cluster_statistics.iloc[j,0]
                s =cluster_statistics.iloc[j,1]
                flag=1
                break
            if(flag==1):
                if(cluster_distance.iloc[i,1] > (m + 3 * s)):
                    outlier+=1
                    for k in range(len(cluster_statistics)):
                        if(cluster_statistics.index[k]!=cluster_distance.iloc[i,0]):
                            dist = cdist([points[i]], [centroids[k]], 'euclidean')
                            m1 = cluster_statistics.iloc[k,0]
                            s1 = cluster_statistics.iloc[k,1]
                            if(dist <= (m1 + 3 * s1)):
                                count+=1
                                class_overlap.append(cluster_statistics.index[k])
        
    #print(count)  #Number of datapoints misclassified as per k-means
    #print(outlier) #Number of datapoints that also belong to other class
    return [count/(dataset.shape[0] * dataset.shape[1]), outlier/(dataset.shape[0] * dataset.shape[1])],class_overlap
