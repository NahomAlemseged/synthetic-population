from airflow import DAG
from airflow.decorators import task
from airflow.operators.bash import BashOperator
from datetime import datetime
from src.etl.etl import main as etl_main
from src.ingestion.ingest import main as ingest_main
from src.generate.generate import main as generate_main
from src.validate.train import main as train_main

BASE_DIR = "/home/nahom/airflow_projects/synthetic-population"

with DAG(
    dag_id="my_pipeline_taskflow",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,   # correct for manual trigger
    catchup=False,
    tags=["etl", "ml", "synthetic_population"],
) as dag:

    extract = BashOperator(
        task_id="extract",
        bash_command=f"""
        set -e
        echo "============================================"
        echo "STARTING EXTRACT STEP"
        echo "============================================"
        bash {BASE_DIR}/data/raw/extract_load.sh
        echo "============================================"
        echo "EXTRACT STEP COMPLETED"
        echo "============================================"
        """
    )

    @task
    def etl():
        etl_main()

    @task
    def ingest():
        ingest_main()

    @task
    def generate():
        generate_main()

    @task
    def train():
        train_main()

    extract >> etl() >> ingest() >> generate() >> train()
