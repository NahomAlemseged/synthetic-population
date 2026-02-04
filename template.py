import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

list_of_files = [
    ".gitignore",

    # Source code
    "src/__init__.py",
    "src/main.py",

    # ETL
    "src/etl/__init__.py",
    "src/etl/etl.py",

    # Generate
    "src/generate/__init__.py",
    "src/generate/generate.py",

    # Validate
    "src/validate/__init__.py",
    "src/validate/train.py",
    "src/validate/evaluate.py",

    # Utils
    "src/utils/__init__.py",
    "src/utils/helpers.py",
    "src/utils/logging.py",
    "src/utils/exceptions.py",

    # Data folders
    "data/raw/",
    "data/bronze/",
    "data/transformed/",
    "data/gold/",

    # Airflow DAGs
    "dags/__init__.py",
    "dags/pipeline.py",

    # Notebooks
    "notebooks/__init__.py",

    # Assets
    "assets/plots/",
    "assets/reports/",
    "assets/logs/",

    # Config files
    "config/params.yaml",
    "config/db_config.yaml",
    "config/schema/schema.yaml",

    # Others
    "Dockerfile",
    ".github/workflows/",
    "requirements.txt",
    "README.md"
]

for filepath in list_of_files:
    path = Path(filepath)

    # Treat paths ending with '/' as directories
    if filepath.endswith("/"):
        path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory: {path}")
    else:
        # Ensure parent directories exist
        path.parent.mkdir(parents=True, exist_ok=True)
        # Create file if it doesn't exist
        if not path.exists():
            path.touch()
            logging.info(f"Created empty file: {path}")
        else:
            logging.info(f"File already exists: {path}")
