# FeatureSelectionTechniqueRecommendation
This is an application of the Meta-learning ontology where feature selection technique is recommended for the given dataset. This repository contains all the resources needed to build a feature selection technique recommendation system.


#### metaFeatureExtraction.ipynb

**Purpose:** Extract meta-features and write them into a file. 
*Input:* Datasets
*Output:* Meta-features in csv format
Dependencies: file_operations.py, pre_processing.py, simple_characteristics.py, quality_metrics.py, information_theoretic.py, arfftocsv.py


#### ensemble_final.ipynb

**Purpose:** Generate knowledge base
*Input:* Dataests
*Output:* Performance metrics for each dataset with feature selection techniques. 
Dependencies: file_operations.py, pre_processing.py, simple_characteristics.py, arfftocsv.py, CFS.py, fc.py, multisurf.py, chiSquare.py, testrelief.py, relieff.py


#### creatingBins.ipynb

**Purpose:** Generating bins and normalization limits
*Input:* Meta features and performance metrics
*Output:* NormalizationValues and BinValues, Limits
Dependencies: postProcessing.py


#### optimalalphabeta_featureselectiontechnique.ipynb

**Purpose:** Generating a robust knoweldge base
*Input:* Meta features, Perfomance metrics, Binned features, Normalised values
*Output:* Identification of robust knowledge base


#### recommendation.ipynb

**Purpose:** Comparing accuracy of various models without modifynig the dataset.
*Input:* Knowledge base
*Output:* Perfomance metrics


#### sampling.ipynb

**Purpose:** Comparing the accuracy of various models without modifying the dataset.
*Input:* Knowledge base
*Output:* Performance metrics


#### sampling_removedleastoccurances.ipynb

**Purpose:** Comparing the accuracy of various models without modifying the dataset.
*Input:* Knowledge base
*Output:* Performance metrics


#### withoutFS.ipynb

**Purpose:** To check the impact of Feature selection techniques on ML classification performance
*Input:* Datasets
*Output:* Statistical significance


#### statisticalAnalysis.ipynb

**Purpose:** All the graphs required in the thesis


#### LoadingTriples.ipynb

**Purpose:** Converting tabular data into RDF format and uploading into Triplestore. 
*Input:* Knowledge base
*Output:* Data in RDF format and in the fuseki server.
