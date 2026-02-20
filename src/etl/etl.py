import os
from pathlib import Path
import yaml
import torch

# ==============================
# GPU Detection
# ==============================

GPU_AVAILABLE = torch.cuda.is_available()

if GPU_AVAILABLE:
    print("🚀 GPU detected — using RAPIDS")
    import dask_cudf as dd
else:
    print("💻 Using CPU Dask")
    import dask.dataframe as dd

# ==============================
# Load Config
# ==============================

CONFIG_PATH = Path("/content/synthetic-population_/config/params.yaml")

with open(CONFIG_PATH, "r") as f:
    params_ = yaml.safe_load(f)

# ==============================
# ETL Class
# ==============================

class ETL:

    def __init__(self):
        self.input_paths = params_["etl"]["input"]
        self.output_path = Path(params_["etl"]["output"][0])

    # --------------------------------------------------
    # EXTRACT + TRANSFORM
    # --------------------------------------------------

    def extract_transform(self):
        """
        Scan directories, load BASE_DATA_1 and GROUPER txt files,
        and return a dictionary of Dask (or cuDF) DataFrames
        """
        dfs = {}

        for dir_path in self.input_paths:
            dir_path = Path(dir_path)
            print(f"\n📂 Scanning {dir_path}")

            if not dir_path.exists():
                print("❌ Directory does not exist")
                continue

            txt_files = list(dir_path.rglob("*.txt"))
            print(f"Found {len(txt_files)} txt files")

            # Filter files based on THCIC naming
            if "outpatient" in dir_path.name.lower():
                base_files = [str(p) for p in txt_files if "BASE" in p.name]
                grouper_files = [str(p) for p in txt_files if "GROUPER" in p.name]
            else:
                base_files = [str(p) for p in txt_files if "BASE_DATA_1" in p.name]
                grouper_files = [str(p) for p in txt_files if "GROUPER" in p.name]

            # ----------------------
            # BASE FILES
            # ----------------------
            if base_files:
                print(f"Loading {len(base_files)} BASE files")
                # Dask can read multiple files directly
                df_base = dd.read_csv(base_files, sep="\t", dtype=str)
                df_base["TYPE"] = dir_path.name
                dfs[f"df_base_1_{dir_path.name}"] = df_base
                print(f"✅ df_base_1_{dir_path.name} loaded")
            else:
                print("⚠️ No BASE files found")

            # ----------------------
            # GROUPER FILES
            # ----------------------
            if grouper_files:
                print(f"Loading {len(grouper_files)} GROUPER files")
                df_grouper = dd.read_csv(grouper_files, sep="\t", dtype=str)
                df_grouper["TYPE"] = dir_path.name
                dfs[f"df_grouper_{dir_path.name}"] = df_grouper
                print(f"✅ df_grouper_{dir_path.name} loaded")
            else:
                print("⚠️ No GROUPER files found")

        if not dfs:
            raise ValueError("❌ No datasets extracted in ETL.")

        return dfs

    # --------------------------------------------------
    # LOAD (SAVE CLEAN DATA)
    # --------------------------------------------------

    def load(self, dfs):
        """
        Save extracted DataFrames to output path in Parquet format
        """
        self.output_path.mkdir(parents=True, exist_ok=True)

        for dataset_name, ddf in dfs.items():
            out_dir = self.output_path / dataset_name
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n💾 Saving {dataset_name} → {out_dir}")
            ddf.to_parquet(out_dir, write_index=False)
            print(f"✅ Saved → {out_dir}")

        print("\n🎯 ETL COMPLETE")

# ==============================
# RUN
# ==============================

def main():
    """
    Run the ETL pipeline sequentially (safe for GPU)
    """
    etl = ETL()
    dfs = etl.extract_transform()
    etl.load(dfs)
    print("\n🎯 ETL pipeline finished successfully!")

if __name__ == "__main__":
    main()
