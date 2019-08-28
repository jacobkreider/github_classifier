from google.cloud import bigquery
from google.oauth2 import service_account
from google.cloud import bigquery_storage_v1beta1
from sklearn import preprocessing
from sklearn.cluster import KMeans
import pandas as pd

import pandas_gbq
import random

credentials = service_account.Credentials.from_service_account_file(
    'main-credentials.json')
project_id = 'poach-easy-2'
client = bigquery.Client(credentials= credentials, project=project_id)
bigquery_dataset = client.dataset('github_project')
labeled_data_ref = bigquery_dataset.table('labeled_data_dev')

bqstorageclient = bigquery_storage_v1beta1.BigQueryStorageClient(
    credentials=credentials
)

kmeans_source_query = """
        SELECT * FROM
        github_project.kmeans_data_dev
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

def salary_estimate(row):
    if row['cluster'] == 1:
        val = random.randrange(60000, 85000, 500)
    elif row['cluster'] == 2:
        val = random.randrange(90000, 120000, 500)
    elif row['cluster'] == 3:
        val = random.randrange(75000, 90000, 500)
    elif row['cluster'] == 4:
        val = random.randrange(70000, 100000, 500)
    else:
        val = 1
    return val


labeled_data = unlabeled_data
labeled_data['Cluster_Name'] = labeled_data.apply(cluster_names, axis=1)
print('Data successfully labeled')

labeled_data['Estimated_Salary'] = labeled_data.apply(salary_estimate, axis=1)
print("Salary Estimates successfully generated")

pandas_gbq.to_gbq(labeled_data, 'github_project.labeled_data_dev', project_id=project_id, if_exists='replace')
# client.load_table_from_dataframe(labeled_data, labeled_data_ref).result()
"""
labeled_data_1 = labeled_data.sample(frac=0.1, replace=False, random_state=42)
labeled_data_2a = labeled_data.drop(labeled_data_1.index)

labeled_data_2 = labeled_data_2a.sample(frac=0.1, replace=False, random_state=42)
labeled_data_3a = labeled_data_2a.drop(labeled_data_2.index)

labeled_data_3 = labeled_data_3a.sample(frac=0.1, replace=False, random_state=42)
labeled_data_4a = labeled_data_3a.drop(labeled_data_3.index)

labeled_data_4 = labeled_data_4a.sample(frac=0.1, replace=False, random_state=42)
labeled_data_5a = labeled_data_4a.drop(labeled_data_4.index)

labeled_data_5 = labeled_data_5a.sample(frac=0.1, replace=False, random_state=42)
labeled_data_6a = labeled_data_5a.drop(labeled_data_5.index)

labeled_data_6 = labeled_data_6a.sample(frac=0.1, replace=False, random_state=42)
labeled_data_7a = labeled_data_6a.drop(labeled_data_6.index)

labeled_data_7 = labeled_data_7a.sample(frac=0.1, replace=False, random_state=42)
labeled_data_8a = labeled_data_7a.drop(labeled_data_7.index)

labeled_data_8 = labeled_data_8a.sample(frac=0.1, replace=False, random_state=42)
labeled_data_9a = labeled_data_8a.drop(labeled_data_8.index)

labeled_data_9 = labeled_data_9a.sample(frac=0.1, replace=False, random_state=42)
labeled_data_10 = labeled_data_9a.drop(labeled_data_9.index)

labeled_data_1.to_csv('labeled_data_1.csv')
labeled_data_2.to_csv('labeled_data_2.csv')
labeled_data_3.to_csv('labeled_data_3.csv')
labeled_data_4.to_csv('labeled_data_4.csv')
labeled_data_5.to_csv('labeled_data_5.csv')
labeled_data_6.to_csv('labeled_data_6.csv')
labeled_data_7.to_csv('labeled_data_7.csv')
labeled_data_8.to_csv('labeled_data_8.csv')
labeled_data_9.to_csv('labeled_data_9.csv')
labeled_data_10.to_csv('labeled_data_10.csv')
"""
print('Data successfully transferred to BigQuery')
print('Exiting program now')




