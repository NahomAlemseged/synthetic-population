import subprocess
from datetime import datetime
import os

from src.etl.etl import main as etl_main
from src.ingestion.ingest import main as ingest_main
from src.generate.generate import main as generate_main
from src.validate.train import main as train_main
from src.validate.evaluate import main as evaluate_main

BASE_DIR = os.path.abspath(os.getcwd()) 



def extract():
    print("============================================")
    print("STARTING EXTRACT STEP")
    print("============================================")

    subprocess.run(
        ["bash", f"/content/drive/MyDrive/data_THCIC/extract_load.sh"],/extract_load.sh"],
        check=True
    )

    print("============================================")
    print("EXTRACT STEP COMPLETED")
    print("============================================")


def main():
    start_time = datetime.now()
    print(f"PIPELINE STARTED AT {start_time}")

    extract()
    etl_main()
    ingest_main()
    generate_main()
    train_main()
    evaluate_main()

    end_time = datetime.now()
    print(f"PIPELINE FINISHED AT {end_time}")
    print(f"TOTAL RUNTIME: {end_time - start_time}")


if __name__ == "__main__":
    main()

### Instructions to run:
# python -m src.main --n_samples 100000 --epochs 10 --num_processes 4 --sample_rows 100000
