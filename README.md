# 

<h1 align="center">Prediction of customer preferences based on its social information (Use case : cars categories) </h1>
<h2 align="center">we know what's best for you</h2>
<p align="center">
<img src ="https://github.com/BentarHamza/ClientCarPrediction/blob/main/photo/big-data-automotive-industry-header-1.jpg">
</p>

> __Note__
> About dataset.

> Data are giving to me in a contexte of academic project simulating a car dealer provides us the folowing dataset in order to help him better target the vehicles likely to interest his customers. For this purpose and to create a Datalake in the objective of Storing data from operational systems and automating this analysis: 
>- Its catalog of vehicles
>- Its file concerning the purchases of its customers of the current year
> - An access to all the information on the registrations made this year

<h1> $\textcolor{brown}{\text{I.1 Machine Learning for predictions}}$ </h1>

<h2> $\textcolor{Orange}{\text{I.1 cleaning and exploratory data analysis}}$ </h2>
In this part we begin by identifying and correcting our data from duplicates, inconsistent, erroneous or wrong data in order to evoid unaccurate or misleading conclusions in our analysis.
<h3> $\textcolor{BurntOrange}{\text{I.1.1 Presenting some problems in DataSet}}$ </h3>

<div align="center">
  <img src="https://github.com/BentarHamza/ClientCarPrediction/blob/main/photo/Capture.PNG" width ="200" height = "200" style="display:inline-block;">
  <img src="https://github.com/BentarHamza/ClientCarPrediction/blob/fa6a357ea6733b2c9d0165fb96790d42e93c8095/photo/Age.PNG" width ="200" height = "200" style="display:inline-block;">
</div>
All duplicated rows in are deleted from our DataSet, categories that represent the same information in categorical columns are unified and all missing values that can be filled by a mean or median value are modified. 

<h3> $\textcolor{BurntOrange}{\text{I.1.2 Analysing our Data}}$ </h3>
And after that we start our exploratory trip inside our dataset.

<h2> $\textcolor{Orange}{\text{I.2 Unsupervised classification of car's categories}}$ </h2>

There is two approachs to classify cars categories : 
- a supervised approach based on business knowledge 
- a unsupervised approach so the model is given a set of unlabeled data, and it must find patterns and structure within the data on its own. 

We are using unsupervised machine learning in our case because our data aren't labeled and we can't contact our client to have his own categorization. 

<h3> $\textcolor{BurntOrange}{\text{I.2.1 Method K-MEANS}}$ </h3>

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

<h3> $\textcolor{BurntOrange}{\text{I.2.2 Agglomerative Clustering  Method}}$ </h3>

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

<h2> $\textcolor{Orange}{\text{I.3 supervised classification of cars categories according to customer profile }}$ </h2>



















