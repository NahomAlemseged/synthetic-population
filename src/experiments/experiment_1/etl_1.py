import pandas as pd
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
# import gower
from sklearn.neighbors import NearestNeighbors
from func import *
from helpers import *
import os
from pathlib import Path
import json
import yaml
import dask.dataframe as dd

# --------------------------
# Load config
# --------------------------
with open('/mnt/c/Users/nahomw/Desktop/from_mac/nahomworku/Desktop/uthealth/gra_project/synthetic-population/config/params.yaml') as f:
    params_ = yaml.safe_load(f)

input_paths = params_['etl']['input']  # list of folders
# output_base = Path(params_['etl']['output'])

# --------------------------
# ETL Class
# --------------------------
class ETL:
    def __init__(self, root_paths, chunksize=100_000):
        self.root_paths = [Path(p) for p in root_paths] if isinstance(root_paths, list) else [Path(root_paths)]
        self.chunksize = chunksize
        # Column names in all caps
        self.keep_columns = [
            'RECORD_ID', 'DISCHARGE', 'TYPE_OF_ADMISSION', 'SOURCE_OF_ADMISSION',
            'PAT_ZIP', 'PAT_COUNTY', 'PUBLIC_HEALTH_REGION', 'PAT_STATUS',
            'SEX_CODE', 'RACE', 'ETHNICITY', 'ADMIT_WEEKDAY', 'LENGTH_OF_STAY',
            'ILLNESS_SEVERITY', 'PAT_AGE', 'FIRST_PAYMENT_SRC', 'APR_MDC'
        ]

    def extract(self):
        """
        Recursively find .txt files containing BASE_1 or GROUPER
        """
        extracted_data = {'emergency': [], 'inpatient': [], 'outpatient': []}

        for root_path in self.root_paths:
            if not root_path.exists():
                print(f"Warning: folder does not exist: {root_path}")
                continue

            for txt_file in root_path.rglob("*.txt"):
                file_path_lower = str(txt_file).lower()
                dataset_type = next((k for k in extracted_data if k in file_path_lower), None)
                if dataset_type is None:
                    print(f"Skipping file (dataset not recognized): {txt_file}")
                    continue

                if "base_1" in file_path_lower:
                    extracted_data[dataset_type].append(("BASE_1", txt_file))
                elif "grouper" in file_path_lower:
                    extracted_data[dataset_type].append(("GROUPER", txt_file))
                else:
                    continue

                print(f"Extracted file for {dataset_type}: {txt_file}")

        return extracted_data

    def transform(self, extracted_data):
        """
        Merge BASE_1 and GROUPER files per dataset type into DataFrames
        """
        merged_dfs = {}

        for dataset_type, files in extracted_data.items():
            base_files = [f for t, f in files if t == "BASE_1"]
            grouper_files = [f for t, f in files if t == "GROUPER"]

            # Read and merge BASE_1 files
            base_df = pd.DataFrame()
            for f in base_files:
                for chunk in pd.read_csv(f, sep='\t', dtype=str, chunksize=self.chunksize):
                    base_df = pd.concat([base_df, chunk], ignore_index=True)

            # Read and merge GROUPER files
            grouper_df = pd.DataFrame()
            for f in grouper_files:
                for chunk in pd.read_csv(f, sep='\t', dtype=str, chunksize=self.chunksize):
                    grouper_df = pd.concat([grouper_df, chunk], ignore_index=True)

            # Merge BASE + GROUPER
            if not base_df.empty and not grouper_df.empty and 'RECORD_ID' in base_df.columns and 'RECORD_ID' in grouper_df.columns:
                merged = pd.merge(base_df, grouper_df, on='RECORD_ID', how='outer', suffixes=("_BASE", "_GROUPER"))
            else:
                merged = pd.concat([base_df, grouper_df], ignore_index=True)

            # Keep only desired columns in ALL CAPS
            merged = merged[[c for c in self.keep_columns if c in merged.columns]]
            merged.dropna(inplace=True)

            merged_dfs[dataset_type] = merged
            print(f"{dataset_type}: transformed shape = {merged.shape}")

        return merged_dfs

    def load(self, merged_dfs):
        """
        Save each merged dataset as a CSV file
        """
        for dataset_type, df in merged_dfs.items():
            out_dir = output_base / dataset_type / "output"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{dataset_type}.csv"
            df.to_csv(out_file, index=False)
            print(f"Saved {dataset_type} → {out_file}")

# --------------------------
# Run ETL
# --------------------------
if __name__ == "__main__":
    etl = ETL(root_paths=input_paths, chunksize=100_000)
    extracted = etl.extract()
    merged_transformed = etl.transform(extracted)
    etl.load(merged_transformed)
    print("✅ ETL finished. All datasets merged and saved as CSV.")
