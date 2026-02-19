import sys
import os
from pathlib import Path
import yaml
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

# --------------------------
# Load config
# --------------------------
with open('/content/synthetic-population_/config/params.yaml') as f:
    params_ = yaml.safe_load(f)

# --------------------------
# ETL Class (Parallelized)
# --------------------------
class ETL:
    def __init__(self):
        self.input_path = params_['etl']['input']  # list of folders
        self.output_path = params_['etl']['output']  # list or str path

        # Start a Dask cluster for parallel processing
        self.cluster = LocalCluster(n_workers=os.cpu_count(), threads_per_worker=1)
        self.client = Client(self.cluster)
        print(f"✅ Dask cluster started with {os.cpu_count()} workers")

    def extract_transform(self):
        """
        Extract and transform all files in parallel using Dask
        """
        dfs = {}

        for i in range(len(self.input_path)):
            dir_path = Path(self.input_path[i])

            # Find files
            file_grouper = [str(p) for p in dir_path.rglob("*.txt") if "IP_ED_GROUPER_1q2023_tab.txt" in p.name]
            file_base = [str(p) for p in dir_path.rglob("*.txt") if "IP_ED_BASE_DATA_1_1q2023_" in p.name]

            # Grouper files
            if file_grouper:
                df_grouper = dd.concat(
                    [dd.read_csv(fp, sep="\t", dtype=str) for fp in file_grouper],
                    ignore_index=True, axis=0, interleave_partitions=True
                )
                df_grouper['TYPE'] = dir_path.name
                dfs[f"df_grouper_{dir_path.name}"] = df_grouper
                print(f"..df_grouper_{dir_path.name} loaded and merged..")
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
                print(f"..df_base_1_{dir_path.name} loaded and merged..")
            else:
                print(f"..No base files found in {dir_path.name}, skipping..")

        return dfs

    def load(self, dfs):
        """
        Save Dask DataFrames in parallel to CSVs
        """
        try:
            for dataset_type, ddf in dfs.items():
                out_dir = Path(self.output_path[0]) / dataset_type
                out_dir.mkdir(parents=True, exist_ok=True)

                csv_file = out_dir / f"{dataset_type}.csv"
                print(f"Saving → {csv_file}")

                # Use Dask to save in parallel (single CSV per dataset)
                ddf.repartition(npartitions=os.cpu_count()).to_csv(
                    csv_file,
                    index=False,
                    single_file=True
                )
                print(f"✅ Saved {dataset_type} → {csv_file}")

            print(f"Successfully saved all datasets to {self.output_path}")

        except Exception as e:
            print(f"❌ Error in load: {e}")

        finally:
            print("Exiting Loading stage")
            # Shutdown Dask client
            self.client.close()
            self.cluster.close()
            print("✅ Dask cluster closed")


# --------------------------
# Run ETL
# --------------------------
def main():
    etl = ETL()
    extracted = etl.extract_transform()
    etl.load(extracted)
    print("✅ ETL finished. All datasets merged and saved as CSV.")


if __name__ == "__main__":
    main()
