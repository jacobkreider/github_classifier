"""Contains function designed to pull BigQuery data into format for classifier"""
import pandas as pd
import numpy as np


def generate_data(client, bqstorageclient):

    """Returns full data, ready for splitting and clustering"""
    language_query = """
        SELECT repo_name, Language, Bytes,
        CASE 
        when LOWER(Language) NOT IN ('javascript', 'python', 'ruby', 'java'
                                    , 'php', 'c++', 'css', 'c#', 'go'
                                    , 'c', 'typescript', 'shell', 'swift'
                                    , 'scala', 'objective-c') then 'Other'
        else Language END AS language_category
        FROM github_project.languages    
        ORDER BY repo_name
        
        
        """

    language_query_dataframe = (
        client.query(language_query)
        .result()
        .to_dataframe(bqstorage_client=bqstorageclient)
    )

    commit_query = """
        SELECT Commit, Author, repo as repo_name
        FROM github_project.commits
        ORDER BY repo
        LIMIT 10000000
        """

    commit_query_dataframe = (
        client.query(commit_query)
        .result()
        .to_dataframe(bqstorage_client=bqstorageclient)
    )

    merged_data = pd.merge(commit_query_dataframe, language_query_dataframe, on='repo_name')

    main_cluster_data = merged_data.groupby('Author').agg({'repo_name': 'nunique', 'Commit': 'sum'
                                                          , 'language_category': 'nunique'})

    language_pivot = pd.pivot_table(merged_data, values='Commit', index=['Author']
                                , columns=['language_category'], aggfunc=np.sum
                                , fill_value=0).reset_index()

    final_cluster_data = pd.merge(main_cluster_data, language_pivot, on='Author')

    return final_cluster_data





