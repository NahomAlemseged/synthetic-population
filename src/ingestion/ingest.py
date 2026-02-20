import yaml
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# ==============================
# Load Config
# ==============================
CONFIG_PATH = Path("/content/synthetic-population_/config/params.yaml")

with open(CONFIG_PATH, "r") as f:
    params_ = yaml.safe_load(f)

input_dir = Path(params_["ingestion"]["input_dir"])
output_path = Path(params_["ingestion"]["output"])
test_size = params_["ingestion"].get("test_size", 0.2)
random_state = params_["ingestion"].get("random_state", 42)

# ==============================
# Ingestion Class
# ==============================
class Ingestion:
    def __init__(self, input_dir, output_path):
        self.input_dir = Path(input_dir)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def ingest_data(self):
        """
        Merge all BASE and GROUPER folders automatically
        """
        output_file = self.output_path / "final_data.csv"
        first_file = True

        # Automatically find base/grouper pairs
        base_folders = sorted(self.input_dir.glob("df_base_1_*"))
        grouper_folders = sorted(self.input_dir.glob("df_grouper_*"))

        # Ensure matching datasets
        for base_folder in base_folders:
            name_suffix = base_folder.name.replace("df_base_1_", "")
            matching_grouper = self.input_dir / f"df_grouper_{name_suffix}"

            if not matching_grouper.exists():
                print(f"⚠️ No grouper folder found for {base_folder.name}, skipping")
                continue

            # Merge all files in the folder
            base_files = sorted(base_folder.glob("*.parquet"))
            grouper_files = sorted(matching_grouper.glob("*.parquet"))

            for bf, gf in zip(base_files, grouper_files):
                print(f"\n📂 Processing {bf.name} + {gf.name}")
                df_base = pd.read_parquet(bf)
                df_base = pd.read_csv(bf, dtype=str, usecols=[
                    'RECORD_ID', 'DISCHARGE', 'TYPE_OF_ADMISSION', 'SOURCE_OF_ADMISSION',
                    'PAT_ZIP', 'PAT_COUNTY', 'PUBLIC_HEALTH_REGION', 'PAT_STATUS',
                    'SEX_CODE', 'RACE', 'ETHNICITY', 'ADMIT_WEEKDAY', 'LENGTH_OF_STAY',
                    'PAT_AGE', 'FIRST_PAYMENT_SRC'
                ])
                df_grouper = pd.read_csv(gf, dtype=str, usecols=['RECORD_ID','APR_MDC'])
                    # df_grouper.dropna(inplace=True)
                print(f"...shape of grouper data :{df_grouper.shape}")

                df_merged = df_base.merge(df_grouper, on="RECORD_ID", how="inner")
                print(f"...shape of merged data :{df_merged.shape}")
                df_merged.drop(columns=["RECORD_ID"], inplace=True)
                print(f"Merged shape: {df_merged.shape}")

                if first_file:
                    df_merged.to_csv(output_file, index=False, mode='w')
                    first_file = False
                else:
                    df_merged.to_csv(output_file, index=False, mode='a', header=False)

        if first_file:
            raise FileNotFoundError("❌ No data was merged. final_data.csv was not created.")

        print(f"\n✅ Data merged successfully → {output_file}")
        return output_file

    def save_splits(self, csv_file):
        """
        Split into train/test
        """
        print("\n✂️ Splitting train/test...")
        df = pd.read_csv(csv_file)

        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )

        train_file = self.output_path / "train.csv"
        test_file = self.output_path / "test.csv"

        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)

        print(f"✅ Train saved: {train_file} ({len(train_df)} rows)")
        print(f"✅ Test saved: {test_file} ({len(test_df)} rows)")

# ==============================
# Main
# ==============================
def main():
    ingest = Ingestion(input_dir, output_path)
    final_csv = ingest.ingest_data()
    ingest.save_splits(final_csv)
    print("\n🎯 Pipeline Complete!")

if __name__ == "__main__":
    main()
