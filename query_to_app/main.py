
import concurrent.futures

import flask
from google.cloud import bigquery
from google.oauth2 import service_account


app = flask.Flask(__name__)
credentials = service_account.Credentials.from_service_account_file(
    'main-credentials.json')
project_id = 'poach-easy-2'
bigquery_client = bigquery.Client(credentials= credentials, project=project_id)


@app.route("/")
def main():
    
    query_job = bigquery_client.query(
        """
        SELECT *
        FROM (
            SELECT author, repo_count, commits, languages, cluster
                , Cluster_Name, Estimated_Salary, Rank()
                over (Partition BY Cluster_Name
                ORDER BY commits DESC, repo_count  DESC ) AS Rank
            FROM(

                SELECT author, repo_count, commits, languages, cluster
                    , Cluster_Name, Estimated_Salary, row_number()
                    over (Partition BY author 
                    ORDER BY Estimated_Salary DESC) as row_num
                FROM github_project.labeled_data_dev)
                WHERE row_num = 1)
        WHERE Rank <= 10 AND Cluster_Name NOT IN ("MISSED SOMETHING"
        , "Inexperienced Beginner")
        
    """
    )

    return flask.redirect(
        flask.url_for(
            "results",
            project_id=query_job.project,
            job_id=query_job.job_id,
            location=query_job.location,
        )
    )



@app.route("/results")
def results():
    project_id = flask.request.args.get("project_id")
    job_id = flask.request.args.get("job_id")
    location = flask.request.args.get("location")

    query_job = bigquery_client.get_job(
        job_id,
        project=project_id,
        location=location,
    )

    try:
        # Set a timeout because queries could take longer than one minute.
        results = query_job.result(timeout=30)
    except concurrent.futures.TimeoutError:
        return flask.render_template("timeout.html", job_id=query_job.job_id)

    return flask.render_template("query_result.html", results=results)


if __name__ == "__main__":
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host="127.0.0.1", port=8080, debug=True)
# [END gae_python37_bigquery]
