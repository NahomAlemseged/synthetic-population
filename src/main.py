import subprocess
from datetime import datetime
import os
from concurrent.futures import ProcessPoolExecutor

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
        ["bash", "/content/drive/MyDrive/data_THCIC/extract_load.sh"],
        check=True
    )

    print("============================================")
    print("EXTRACT STEP COMPLETED")
    print("============================================")


def run_cpu_stage(func, name):
    print(f"🚀 Starting {name}")
    func()
    print(f"✅ Finished {name}")


def main():
    start_time = datetime.now()
    print(f"PIPELINE STARTED AT {start_time}")

    # 1️⃣ Extract (blocking)
    extract()

    # 2️⃣ Parallel CPU stages
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        futures.append(executor.submit(run_cpu_stage, etl_main, "ETL"))
        futures.append(executor.submit(run_cpu_stage, ingest_main, "INGEST"))

        for f in futures:
            f.result()  # wait

    # 3️⃣ GPU stages (sequential!)
    print("🔥 Starting GENERATE (GPU)")
    generate_main()

    print("🔥 Starting TRAIN (GPU)")
    train_main()

    # 4️⃣ Evaluation (CPU again)
    print("📊 Starting EVALUATE")
    evaluate_main()

    end_time = datetime.now()
    print(f"PIPELINE FINISHED AT {end_time}")
    print(f"TOTAL RUNTIME: {end_time - start_time}")


if __name__ == "__main__":
    main()
