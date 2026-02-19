import os
from pathlib import Path
import yaml
import torch

# --------------------------
# Detect GPU
# --------------------------
GPU_AVAILABLE = torch.cuda.is_available()

if GPU_AVAILABLE:
    print("🚀 GPU detected — using RAPIDS (dask-cudf)")
    import dask_cudf as dd
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
else:
    print("💻 No GPU detected — using Dask CPU")
    import dask.dataframe as dd
    from dask.distributed import Client, LocalCluster

# --------------------------
# Load config
# --------------------------
with open('/content/synthetic-population_/config/params.yaml') as f:
    params_ = yaml.safe_load(f)


class ETL:

    def __init__(self):

        self.input_path = params_['etl']['input']
        self.output_path = params_['etl']['output']

        if GPU_AVAILABLE:
            # 🔥 One worker per GPU
            self.cluster = LocalCUDACluster()
        else:
            # 🔥 Controlled CPU parallelism
            n_workers = min(8, os.cpu_count())
            self.cluster = LocalCluster(
                n_workers=n_workers,
                threads_per_worker=2,
                memory_limit="auto"
            )

        self.client = Client(self.cluster)
        print("✅ Dask cluster started")

    def extract_transform(self):

        dfs = {}

        for dir_path in self.input_path:
            dir_path = Path(dir_path)

            grouper_pattern = str(dir_path / "*GROUPER*.txt")
            base_pattern = str(dir_path / "*BASE_DATA_1_*.txt")

            # --------------------------
            # Grouper
            # --------------------------
            try:
                df_grouper = dd.read_csv(
                    grouper_pattern,
                    sep="\t",
                    dtype=str,
                    blocksize="128MB"
                )
                df_grouper["TYPE"] = dir_path.name
                dfs[f"df_grouper_{dir_path.name}"] = df_grouper
                print(f"✅ Loaded grouper {dir_path.name}")
            except Exception:
                print(f"..No grouper files in {dir_path.name}")

            # --------------------------
            # Base
            # --------------------------
            try:
                df_base = dd.read_csv(
                    base_pattern,
                    sep="\t",
                    dtype=str,
                    blocksize="128MB"
                )
                df_base["TYPE"] = dir_path.name
                dfs[f"df_base_{dir_path.name}"] = df_base
                print(f"✅ Loaded base {dir_path.name}")
            except Exception:
                print(f"..No base files in {dir_path.name}")

        return dfs

    def load(self, dfs):

        for dataset_type, ddf in dfs.items():

            out_dir = Path(self.output_path[0]) / dataset_type
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"💾 Saving {dataset_type} → Parquet")

            # 🔥 Write parallel parquet (fastest + scalable)
            ddf.to_parquet(
                out_dir,
                write_index=False
            )

            print(f"✅ Saved {dataset_type}")

        print("✅ All datasets saved")

        self.client.close()
        self.cluster.close()
        print("✅ Cluster closed")


def main():
    etl = ETL()
    extracted = etl.extract_transform()
    etl.load(extracted)
    print("🎯 ETL finished successfully")


if __name__ == "__main__":
    main()
