# FeatureSelectionTechniqueRecommendation
This is an application of the Meta-learning ontology where a feature selection technique is recommended for the given dataset. This repository contains all the resources needed to build a feature selection technique recommendation system.


#### metaFeatureExtraction.ipynb 

**Purpose:** Extract meta-features and write them into a file. <br>
*Input:* Datasets <br>
*Output:* Meta-features in csv format <br>
Dependencies: file_operations.py, pre_processing.py, simple_characteristics.py, quality_metrics.py, information_theoretic.py, arfftocsv.py <br>


#### ensemble_final.ipynb

**Purpose:** Generate knowledge base <br>
*Input:* Datasets <br>
*Output:* Performance metrics for each dataset with feature selection techniques. <br>
Dependencies: file_operations.py, pre_processing.py, simple_characteristics.py, arfftocsv.py, CFS.py, fc.py, multisurf.py, chiSquare.py, testrelief.py, relieff.py


#### creatingBins.ipynb

**Purpose:** Generating bins and normalization limits <br>
*Input:* Meta features and performance metrics <br>
*Output:* NormalizationValues and BinValues, Limits <br>
Dependencies: postProcessing.py 


#### optimalalphabeta_featureselectiontechnique.ipynb

**Purpose:** Generating a robust knowledge base <br>
*Input:* Meta features, performance metrics, Binned features, Normalised values <br>
*Output:* Identification of robust knowledge base


#### recommendation.ipynb

**Purpose:** Comparing accuracy of various models without modifynig the dataset. <br>
*Input:* Knowledge base <br>
*Output:* Performance metrics 


#### sampling.ipynb

**Purpose:** Comparing the accuracy of various models without modifying the dataset. <br>
*Input:* Knowledge base <br>
*Output:* Performance metrics


#### sampling_removedleastoccurances.ipynb

**Purpose:** Comparing the accuracy of various models without modifying the dataset. <br>
*Input:* Knowledge base <br>
*Output:* Performance metrics


#### withoutFS.ipynb

**Purpose:** To check the impact of Feature selection techniques on ML classification performance. <br>
*Input:* Datasets <br>
*Output:* Statistical significance


#### statisticalAnalysis.ipynb

**Purpose:** All the graphs required in the thesis


#### LoadingTriples.ipynb

**Purpose:** Convert tabular data into RDF format and upload it to a Triplestore. <br>
*Input:* Knowledge base <br>
*Output:* Data is in RDF format and stored in the Fuseki server.
