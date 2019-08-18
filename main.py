from google.cloud import bigquery
from google.oauth2 import service_account
from google.cloud import bigquery_storage_v1beta1
from sklearn.model_selection import train_test_split
import knn_data_create as kdc

credentials = service_account.Credentials.from_service_account_file(
    'msds434-8ba0bd83467d.json')
project_id = 'msds434'
client = bigquery.Client(credentials= credentials, project=project_id)

bqstorageclient = bigquery_storage_v1beta1.BigQueryStorageClient(
    credentials=credentials
)

data_test = kdc.generate_data(client, bqstorageclient)

