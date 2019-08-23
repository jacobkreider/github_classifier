from google.cloud import bigquery
from google.oauth2 import service_account
from google.cloud import bigquery_storage_v1beta1
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.compose import ColumnTransformer
import knn_data_create as kdc
import numpy as np
import pandas_gbq

credentials = service_account.Credentials.from_service_account_file(
    'credential-2.json')
project_id = 'poach-easy-2019'
client = bigquery.Client(credentials= credentials, project=project_id)
bigquery_dataset = client.dataset('github_project')
labeled_data_ref = bigquery_dataset.table('labeled_data_dev')

bqstorageclient = bigquery_storage_v1beta1.BigQueryStorageClient(
    credentials=credentials
)

kmeans_source_query = """
        SELECT * FROM
        github_project.kmeans_data
        """

unlabeled_data = (
        client.query(kmeans_source_query)
        .result()
        .to_dataframe(bqstorage_client=bqstorageclient)
    )

unlabeled_data = unlabeled_data[unlabeled_data['commits'] < 100000]

# Preprocess data for clustering
unlabeled_standardized = unlabeled_data.copy()
col_names = ['repo_count'
                                                  , 'commits', 'languages'
                                                  , 'c', 'csharp', 'cplusplus'
                                                  , 'css', 'go','java'
                                                  ,  'javascript', 'objectivec'
                                                  ,  'other', 'php',  'python'
                                                  ,'ruby',  'scala',  'shell'
                                                  ,  'swift', 'typescript']

features = unlabeled_standardized[col_names]
unlabeled_copy_standardized = preprocessing.scale(features)
unlabeled_copy_standardized = pd.DataFrame(unlabeled_copy_standardized)
unlabeled_standardized[col_names] = unlabeled_copy_standardized

# Create elbow plot to find number of clusters

"""plt.figure(figsize=(10, 8))
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state=42)
    kmeans.fit(unlabeled_copy_standardized)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
"""

# Fitting k-means to the data
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(unlabeled_copy_standardized)

# beginning of  the cluster numbering with 1 instead of 0
y_kmeans1=y_kmeans
y_kmeans1=y_kmeans+1
# New Dataframe called cluster
cluster = pd.DataFrame(y_kmeans1)
# Adding cluster to the Dataset1
unlabeled_data['cluster'] = cluster
# Mean of clusters
kmeans_mean_cluster = pd.DataFrame(round(unlabeled_data.groupby('cluster').mean(), 1))
stats_cluster = pd.DataFrame(unlabeled_data.groupby('cluster').agg({'author':'nunique'}))
print(kmeans_mean_cluster)
print(stats_cluster)




def cluster_names(row):
    if row['cluster'] == 1:
        val = 'Inexperienced Beginner'
    elif row['cluster'] == 2:
        val = 'High Variety and Production, Balanced Language Use'
    elif row['cluster'] == 3:
        val = 'Medium Variety, High Production, Python-Focused'
    elif row['cluster'] == 4:
        val = 'Medium Variety, High Production, JavaScript/CSS Focused'
    else:
        val = 'MISSED SOMETHING'
    return val


labeled_data = unlabeled_data
labeled_data['Cluster_Name'] = labeled_data.apply(cluster_names, axis=1)
print('Data successfully labeled')

# pandas_gbq.to_gbq(labeled_data, 'github_project.labeled_data', project_id=project_id, if_exists='replace')
client.load_table_from_dataframe(labeled_data, labeled_data_ref).result()

print('Data successfully transferred to BigQuery')
print('Exiting program now')




