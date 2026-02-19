import sys
import yaml
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

CONFIG_PATH = Path("/content/synthetic-population_/config/params.yaml")

# Load YAML
with open(CONFIG_PATH, "r") as f:
    params_ = yaml.safe_load(f)

input_paths = params_["ingestion"]["input"]
output_path = Path(params_["ingestion"]["output"])
# test_size = params_["ingestion"].get("test_size", 0.2)
# random_state = params_["ingestion"].get("random_state", 42)

class Ingestion:
    def __init__(self, input_paths, output_path):
        self.input_paths = input_paths
        self.output_path = Path(output_path)

    def ingest_data(self):
        output_path = self.output_path / "final_data.csv"
        self.output_path.mkdir(parents=True, exist_ok=True)

        first_file = True
        for i in range(0, len(self.input_paths), 2):
            base_dir = Path(self.input_paths[i])
            grouper_dir = Path(self.input_paths[i + 1])

            base_files = list(base_dir.rglob("*.csv")) if "base_1" in str(base_dir).lower() else []
            grouper_files = list(grouper_dir.rglob("*.csv")) if "grouper" in str(grouper_dir).lower() else []

            if not base_files or not grouper_files:
                print(f"⚠️ Skipping pair ({base_dir}, {grouper_dir})")
                continue

            for bf in base_files:
                df_base = pd.read_csv(bf, dtype=str, usecols=[
                    'RECORD_ID', 'DISCHARGE', 'TYPE_OF_ADMISSION', 'SOURCE_OF_ADMISSION',
                    'PAT_ZIP', 'PAT_COUNTY', 'PUBLIC_HEALTH_REGION', 'PAT_STATUS',
                    'SEX_CODE', 'RACE', 'ETHNICITY', 'ADMIT_WEEKDAY', 'LENGTH_OF_STAY',
                    'PAT_AGE', 'FIRST_PAYMENT_SRC'
                ])
                print(f"...shape of base data :{df_base.shape}")
                for gf in grouper_files:
                    df_grouper = pd.read_csv(gf, dtype=str, usecols=['RECORD_ID','APR_MDC'])
                    # df_grouper.dropna(inplace=True)
                    print(f"...shape of grouper data :{df_grouper.shape}")

                    df_merged = df_base.merge(df_grouper, on="RECORD_ID")

                    # df_merged.dropna(inplace=True)
                    if first_file:
                        df_merged.to_csv(output_path, index=False, mode='w')
                        first_file = False
                    else:
                        df_merged.to_csv(output_path, index=False, mode='a', header=False)

        print(f"✅ Data merged incrementally → {output_path}")
        return output_path

    def save_splits(self, csv_file):
        print("✂️ Splitting train/test...")
        df = pd.read_csv(csv_file)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        train_file = self.output_path / "train.csv"
        test_file = self.output_path / "test.csv"

        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        print(f"✅ Train: {train_file} ({len(train_df)} rows)")
        print(f"✅ Test: {test_file} ({len(test_df)} rows)")

def main():
    ingest = Ingestion(input_paths, output_path)
    final_csv = ingest.ingest_data()
    ingest.save_splits(final_csv)
    print("🎯 Done!")

# if __name__ == "__main__":
#     main()
