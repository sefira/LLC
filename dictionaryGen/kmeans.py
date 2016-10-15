import pandas as pd
from sklearn.cluster import KMeans

data_filename = "deep_cnn_feature.dat.csv"

deep_feature_data = pd.read_csv(data_filename)

n_dictionary = 3
kmeans_res = KMeans(init='random', n_clusters=n_dictionary, 
                n_init=10).fit(deep_feature_data)
                
#print(deep_feature_data)
#print(kmeans_res.labels_)
print(kmeans_res.inertia_)
#print(kmeans_res.cluster_centers_)

centroids = pd.DataFrame(kmeans_res.cluster_centers_)
print(centroids)

centroids.to_csv("centroids.csv", index = False)

