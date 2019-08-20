from google.cloud import bigquery
from google.oauth2 import service_account
from google.cloud import bigquery_storage_v1beta1
import knn_data_create as kdc
import numpy as np
import pandas_gbq

credentials = service_account.Credentials.from_service_account_file(
    'poach-easy-cred.json')
project_id = 'poach-easy'
bigquery_dataset = 'github_project'
client = bigquery.Client(credentials= credentials, project=project_id)

bqstorageclient = bigquery_storage_v1beta1.BigQueryStorageClient(
    credentials=credentials
)

print('Loaded credentials')


def bq_create_dataset(client, dataset):
    bigquery_client = client
    dataset_ref = bigquery_client.dataset(dataset)

    try:
        bigquery_client.get_dataset(dataset_ref)
    except Exception:
        new_dataset = bigquery.Dataset(dataset_ref)
        new_dataset = bigquery_client.create_dataset(dataset)
        print('Dataset {} created.'.format(new_dataset.dataset_id))


def bq_create_table(client, dataset, tablename, schemaref):
    bigquery_client = client
    dataset_ref = bigquery_client.dataset(dataset)

    # Prepares a reference to the table
    table_ref = dataset_ref.table(tablename)

    try:
        bigquery_client.get_table(table_ref)
    except Exception:
        schema = schemaref
        table = bigquery.Table(table_ref, schema=schema)
        table = bigquery_client.create_table(table)
        print('table {} created.'.format(table.table_id))


print('Functions created')

unlabeled_data_schema = [
            bigquery.SchemaField('author', 'STRING', mode='REQUIRED'),
            bigquery.SchemaField('repo_count', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('commits', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('languages', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('c', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('csharp', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('cplusplus', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('css', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('go', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('java', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('javascript', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('objectivec', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('other', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('php', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('python', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('ruby', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('scala', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('shell', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('swift', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('typescript', 'INTEGER', mode='REQUIRED')

        ]

kmeans_data_schema = [
            bigquery.SchemaField('author', 'STRING', mode='REQUIRED'),
            bigquery.SchemaField('repo_count', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('commits', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('languages', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('c', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('csharp', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('cplusplus', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('css', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('go', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('java', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('javascript', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('objectivec', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('other', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('php', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('python', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('ruby', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('scala', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('shell', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('swift', 'INTEGER', mode='REQUIRED'),
            bigquery.SchemaField('typescript', 'INTEGER', mode='REQUIRED')
    ]

print('Schemas designed')

raw_data_import = kdc.generate_data(client, bqstorageclient)
raw_data_import = raw_data_import.rename(columns={'Author': 'author', 'repo_name': 'repo_count'
                                                  , 'Commit': 'commits', 'language_category': 'languages'
                                                  , 'C': 'c', 'C#': 'csharp', 'C++': 'cplusplus'
                                                  , 'CSS': 'css', 'Go': 'go', 'Java': 'java'
                                                  , 'JavaScript': 'javascript', 'Objective-C': 'objectivec'
                                                  , 'Other': 'other', 'PHP': 'php', 'Python': 'python'
                                                  , 'Ruby': 'ruby', 'Scala': 'scala', 'Shell': 'shell'
                                                  , 'Swift': 'swift', 'TypeScript': 'typescript'})

print('Raw data imported')
print(raw_data_import.head())



kmeans_data = raw_data_import.sample(frac=0.75, replace=False, random_state=42)
unlabeled_data = raw_data_import.drop(kmeans_data.index)

print('Data created and ready')





#bq_create_table(client, bigquery_dataset, unlabeled_data, unlabeled_data_schema)
#bq_create_table(client, bigquery_dataset, kmeans_data, kmeans_data_schema)

pandas_gbq.to_gbq(kmeans_data, 'github_project.kmeans_data', project_id=project_id, if_exists='replace')
pandas_gbq.to_gbq(unlabeled_data, 'github_project.unlabeled_data', project_id=project_id, if_exists='replace')

print('BigQuery tables written successfully!')

print(list(kmeans_data.columns))

print(kmeans_data.head())
print(len(kmeans_data))
print(len(unlabeled_data))
print(np.shape(kmeans_data))
print(np.shape(unlabeled_data))

print('Success. Exiting program now')
