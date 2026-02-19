import sys
import os
import pandas as pd
from pathlib import Path
import yaml
import dask.dataframe as dd
from sqlalchemy import create_engine

# --------------------------
# Load config
# --------------------------
with open('/content/synthetic-population_/config/params.yaml') as f:
    params_ = yaml.safe_load(f)


# --------------------------
# ETL Class
# --------------------------
class ETL:
    def __init__(self):
        self.input_path = params_['etl']['input']  # list of folders
        self.output_path = params_['etl']['output']  # list or str path

    def extract_transform(self):
        dfs = {}
        for i in range(len(self.input_path)):
            dir_path = Path(self.input_path[i])

            # Find files
            file_grouper = [str(p) for p in dir_path.rglob("*.txt") if "IP_ED_GROUPER_1q2023_tab.txt" in p.name]
            # for outpatient, inpatient and emergency
            # file_base = [str(p) for p in dir_path.rglob("*.txt") if "BASE_DATA_1" in p.name or "BASE_1" in p.name] 
            file_base = [str(p) for p in dir_path.rglob("*.txt") if "IP_ED_BASE_DATA_1_1q2023_" in p.name]


            # Grouper files
            if file_grouper:
                df_grouper = dd.concat(
                    [dd.read_csv(fp, sep="\t", dtype=str) for fp in file_grouper],
                    ignore_index=True, axis=0, interleave_partitions=True
                )
                df_grouper['TYPE'] = dir_path.name
                dfs[f"df_grouper_{dir_path.name}"] = df_grouper
                print(f"..df_grouper_{dir_path.name} dataframe loaded and merged..")
            else:
                print(f"..No grouper files found in {dir_path.name}, skipping..")

            # Base files
            if file_base:
                df_base = dd.concat(
                    [dd.read_csv(fp, sep="\t", dtype=str) for fp in file_base],
                    ignore_index=True, axis=0, interleave_partitions=True
                )
                df_base['TYPE'] = dir_path.name
                dfs[f"df_base_1_{dir_path.name}"] = df_base
                print(f"..df_base_1_{dir_path.name} dataframe loaded and merged..")
            else:
                print(f"..No base files found in {dir_path.name}, skipping..")

        return dfs

    def load(self, dfs):
        try:
            for dataset_type, ddf in dfs.items():
                out_dir = Path(self.output_path[0]) / dataset_type
                out_dir.mkdir(parents=True, exist_ok=True)

                csv_file = out_dir / f"{dataset_type}.csv"
                print(f"Saving → {csv_file}")

                # If ddf is a Dask DataFrame
                if hasattr(ddf, "to_csv"):
                    ddf.repartition(npartitions=1).to_csv(
                        csv_file,
                        index=False,
                        single_file=True
                    )
                else:  # Fallback for Pandas DataFrame
                    ddf.to_csv(csv_file, index=False)

                print(f"✅ Saved {dataset_type} → {csv_file}")

            print(f"Successfully saved datasets to {self.output_path}")

        except Exception as e:
            print(f"❌ Error in load: {e}")

        finally:
            print("Exiting Loading stage")


# --------------------------
# Run ETL
# --------------------------
# if __name__ == "__main__":
def main():
    etl = ETL()
    extracted = etl.extract_transform()
    etl.load(extracted)
    print("✅ ETL finished. All datasets merged and saved as CSV.")
