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

input_paths = params_["ingestion"]["input"]
output_path = Path(params_["ingestion"]["output"])


# ==============================
# Ingestion Class
# ==============================

class Ingestion:
    def __init__(self, input_paths, output_path):
        self.input_paths = input_paths
        self.output_path = Path(output_path)

    def ingest_data(self):
        """
        Merge BASE_DATA_1 and GROUPER files
        """
        output_file = self.output_path / "final_data.csv"
        self.output_path.mkdir(parents=True, exist_ok=True)

        first_file = True

        for i in range(0, len(self.input_paths), 2):
            base_dir = Path(self.input_paths[i])
            grouper_dir = Path(self.input_paths[i + 1])

            # Match THCIC naming patterns
            base_files = list(base_dir.rglob("*BASE_DATA_1*.txt"))
            grouper_files = list(grouper_dir.rglob("*GROUPER*.txt"))

            if not base_files or not grouper_files:
                print(f"⚠️ Skipping pair ({base_dir}, {grouper_dir})")
                continue

            print(f"\n📂 Processing:")
            print(f"Base files found: {len(base_files)}")
            print(f"Grouper files found: {len(grouper_files)}")

            for bf in base_files:
                print(f"Reading base: {bf.name}")
                df_base = pd.read_csv(bf, sep="\t", dtype=str)

                for gf in grouper_files:
                    print(f"Reading grouper: {gf.name}")
                    df_grouper = pd.read_csv(gf, sep="\t", dtype=str)

                    # Merge on RECORD_ID
                    df_merged = df_base.merge(df_grouper, on="RECORD_ID", how="inner")

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
        
        if not Path(csv_file).exists():
            raise FileNotFoundError(f"{csv_file} does not exist.")

        df = pd.read_csv(csv_file)

        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42
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
    ingest = Ingestion(input_paths, output_path)
    final_csv = ingest.ingest_data()
    ingest.save_splits(final_csv)
    print("\n🎯 Pipeline Complete!")


if __name__ == "__main__":
    main()
