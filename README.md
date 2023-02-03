# 

<h1 align="center">Prediction of customer preferences based on its social information (Use case : cars categories) </h1>
<h2 align="center">We know what's best for you</h2>
<p align="center">
<img src ="https://github.com/BentarHamza/ClientCarPrediction/blob/main/photo/big-data-automotive-industry-header-1.jpg">
</p>

> __Note__
> About dataset.

> Data are giving to me in a contexte of academic project simulating a car dealer provides us the folowing dataset in order to help him better target the vehicles likely to interest his customers. For this purpose and to create a Datalake in the objective of Storing data from operational systems and automating this analysis: 
>- Its catalog of vehicles
>- Its file concerning the purchases of its customers of the current year
> - An access to all the information on the registrations made this year
----------------------------------
<h1 align="center">Summary </h1>



## I. Machine Learning for predictions  
### I.1 cleaning and exploratory data analysis 
#### I.1.2 Analysing our Data 
### I.2 Unsupervised classification of car's categories 
#### I.2.1 Method K-MEANS 
#### I.2.2 Agglomerative Clustering  Method 
### I.3 Amelioration of our model using other Data (MapReduce USE CASE):
#### I.3.1 Presentation of MapReduce paradigm 
#### I.3.2 Presentation of SPARK APACH Framework 
#### I.3.3 Results 
### I.4 Supervised classification of cars categories according to customer profil 
---------------------------------------





# I. Machine Learning for predictions : 

## I.1 cleaning and exploratory data analysis :
In this part we begin by identifying and correcting our data from duplicates, inconsistent, erroneous or wrong data in order to evoid unaccurate or misleading conclusions in our analysis.
<h3> $\textcolor{BurntOrange}{\text{I.1.1 Presenting some problems in DataSet}}$ </h3>

<div align="center">
  <img src="https://github.com/BentarHamza/ClientCarPrediction/blob/main/photo/Capture.PNG" width ="200" height = "200" style="display:inline-block;">
  <img src="https://github.com/BentarHamza/ClientCarPrediction/blob/fa6a357ea6733b2c9d0165fb96790d42e93c8095/photo/Age.PNG" width ="200" height = "200" style="display:inline-block;">
</div>
All duplicated rows in are deleted from our DataSet, categories that represent the same information in categorical columns are unified and all missing values that can be filled by a mean or median value are modified. 

 ### I.1.2 Analysing our Data :
And after that we start our exploratory trip inside our dataset.

<p align="center">
<img src ="https://github.com/BentarHamza/ClientCarPrediction/blob/main/photo/Financial_capacity.PNG">
</p>



## I.2 Unsupervised classification of car's categories :

There is two approachs to classify cars categories : 
- a supervised approach based on business knowledge 
- a unsupervised approach so the model is given a set of unlabeled data, and it must find patterns and structure within the data on its own. 

We are using unsupervised machine learning in our case because our data aren't labeled and we can't contact our client to have his own categorization. 

### I.2.1 Method K-MEANS :

> __Definition :__
> K-means is a popular unsupervised machine learning algorithm used for clustering. The goal of K-means is to partition a set of data points into K clusters, where each data point belongs to the cluster with the nearest mean. The algorithm works by first randomly initializing K cluster centroids, and then iteratively reassigning each data point to the cluster whose centroid it is closest to, and then recomputing the centroids based on the newly assigned points. This process continues until the cluster assignments no longer change, or a stopping criterion is met. The final clusters represent the partitions of the data into similar groups. K-means is sensitive to the initial centroid values and can have different outcomes depending on the initial centroid placement. Thus it's common to run the algorithm multiple times with different initial centroid values and select the best outcome.

To know optimal number of cluster we use the elbow method as follows : 
<p align="center">
<img src ="https://github.com/BentarHamza/ClientCarPrediction/blob/main/photo/ElbowMethode.PNG">
</p>

> __Interpretation :__
> This graph shows that from 4 clusters the derivative of the function starts to tend towards a fixed value, then we can choose the number of clusters above 4, in our case we limit our clusters to 4 or 5 clusters.

After running kmeans algorithm, we summarize the results (the mean of each cluster) in the following table :  
<p align="center">
<img src ="https://github.com/BentarHamza/ClientCarPrediction/blob/main/photo/kmeans.PNG">
</p>

We try to interpret the result as following : 
Group number with KMEANS  |  interpretability  |
--- | --- |
Cluster 0 | Sedan car (Berline) |
Cluster 1 | Urban car (citadine) |
Cluster 2 | Sport car  |
Cluster 3 | Compact car  |
Cluster 4 | Familial car  |

We use PCA method for visualisation of results, percentage of information retained with the first 2 components: 77.14 %

<p align="center">
<img src ="https://github.com/BentarHamza/ClientCarPrediction/blob/main/photo/KMEANS_PCA.PNG">
</p>

### I.2.2 Agglomerative Clustering  Method :

> __Definition :__
> Agglomerative Clustering is a bottom-up hierarchical clustering technique. It starts by treating each data point as a separate cluster and then merges the most similar clusters together until a stopping criterion is met. The result is a tree-based representation of the clusters, also known as a dendrogram.

We use visual method to find the optimal number of clusters by analysing the dendrogram : 

<p align="center">
<img src ="https://github.com/BentarHamza/ClientCarPrediction/blob/main/photo/dren.PNG">
</p>
The dendrograph shows that the largest vertical distance that does not intersect with any horizontal cluster division line indicates the optimal number of clusters for the Agglomerative Clustering method and in our case it is 4 clusters. But we choose 5 cluster to compare this method to KMEANS. 

After running kmeans algorithm, we summarize the results (the mean of each cluster) in the following table :  


<p align="center">
<img src ="https://github.com/BentarHamza/ClientCarPrediction/blob/main/photo/aglo.PNG">
</p>


We use PCA method for visualisation of results : 


<p align="center">
<img src ="https://github.com/BentarHamza/ClientCarPrediction/blob/main/photo/aglo_PCA.PNG">
</p>

After having tried the two methods of clustering, we choose to continue our work with KMEANS method and we use the model trained with categories data to predict the clusters in the registrations made this year (immatriculation dataset). 

## I.3 improving our model using other DataSet (MapReduce programming model):
### I.3.1 Presentation of MapReduce paradigm : 
MapReduce is a framework for processing parallelizable problems across large datasets using a large number of computers (nodes), collectively referred to as a cluster (if all nodes are on the same local network and use similar hardware) or a grid (if the nodes are shared across geographically and administratively distributed systems, and use more heterogeneous hardware). Processing can occur on data stored either in a filesystem (unstructured) or in a database (structured). MapReduce can take advantage of the locality of data, processing it near the place it is stored in order to minimize communication overhead.

A MapReduce framework (or system) is usually composed of three operations (or steps):

- *Map:* each worker node applies the map function to the local data, and writes the output to a temporary storage. A master node ensures that only one copy of the redundant input data is processed.
- *Shuffle:* worker nodes redistribute data based on the output keys (produced by the map function), such that all data belonging to one key is located on the same worker node.
- *Reduce:* worker nodes now process each group of output data, per key, in parallel.

### I.3.2 Presentation of SPARK APACH Framework : 

<a href="https://spark.apache.org/">SPARK APACH</a> is a MapReduc design and execution framework, Originally (2014) a project of the University of Berkeley, now an open source software from the Apache Foundation.
It was devlopped in *SCALA* (object-oriented language derived from Java and including many aspects of functional languages) and it support many other languages (JAVA, PYTHON, R)
<img src ="https://github.com/BentarHamza/ClientCarPrediction/blob/main/photo/Spark.PNG">

In this work we will use PYSPARK (an interface for Apache Spark in Python. It not only allows you to write Spark applications using Python APIs, but also provides the PySpark shell for interactively analyzing your data in a distributed environment. PySpark supports most of Spark’s features such as Spark SQL, DataFrame, Streaming, MLlib (Machine Learning) and Spark Core).

### I.3.3 Results : 
Configuration and is giving to me by Mr.Sergio SIMONIAN. for more information please enter here .https://github.com/SergioSim/debian-hadoop 
## I.4 supervised classification of cars categories according to customer profil :

> __Definition :__
> 
> __DecisionTreeClassifier__ It's a tree-based approach to model the relationship between the input features and the output labels, in which each internal node represents a test on an input feature and each leaf node represents a class label. The tree is built by recursively splitting the training data based on the feature that results in the largest reduction in impurity. The classifier can be used for both binary and multi-class classification problems and can handle both continuous and categorical input features.
> 
> __RandomForestClassifier__ A random forest is a collection of decision trees, where each tree is trained on a random subset of the data (with replacement) and a random subset of the features. During the prediction, the classifier takes the majority vote of all decision trees' predictions. The random forest classifier can be used for both binary and multi-class classification problems and can handle both continuous and categorical input features.
> 
> __LogisticRegression__ is a variation of logistic regression for multi-class classification problems. It is used to model the probability of an instance belonging to a particular class out of K classes.

We present each model and his accuracy in the table belows : 

Model  |  accuracy |
--- | --- |
DecisionTreeClassifier | 0.780 |
RandomForestClassifier | 0.777 |
LogisticRegression |  0.728 |
MLPClassifier | 0.779  |

The result is giving to marketing department simple of client to predict car's category based on client profile. 

<p align="center">
<img src ="https://github.com/BentarHamza/ClientCarPrediction/blob/main/photo/result.PNG">
</p>

# II. Data LAKE : 

We use in our projet a mixed DATA LAKE architecture  (virtual and physical) built around HIVE consists in relying on external tables to access data from heterogeneous sources (MongoDB, a 2nd NoSQL DB of your choice, Hadoop HDFS, Hadoop Hive) for real-time analysis.  Data access for the data analysis part will be done via the HiveQL language from HIVE.  HiveQL (a kind of Big Data SQL) avoids the physical movement of data in the Data Warehouse database or Data lake.

<img src ="https://github.com/BentarHamza/ClientCarPrediction/blob/main/photo/architecture.PNG">
</p>

> __Note__
> About the Data Lake.
> 
> This part is not completed, the code and clarification will be uploaded after completion. 
