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
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
else:
    print("💻 No GPU detected — using CPU Dask")
    import dask.dataframe as dd
    from dask.distributed import Client, LocalCluster


# ==============================
# Load Config
# ==============================

CONFIG_PATH = Path("/content/synthetic-population_/config/params.yaml")

with open(CONFIG_PATH, "r") as f:
    params_ = yaml.safe_load(f)

input_dirs = params_["etl"]["input"]
output_dir = Path(params_["etl"]["output"][0])


# ==============================
# GPU Pipeline
# ==============================

class GPUPipeline:

    def __init__(self):
        if GPU_AVAILABLE:
            self.cluster = LocalCUDACluster()
        else:
            self.cluster = LocalCluster(
                n_workers=min(8, os.cpu_count()),
                threads_per_worker=2
            )

        self.client = Client(self.cluster)
        print("✅ Dask cluster started")

    # --------------------------------------------------
    # STEP 1: Load TXT Files on GPU
    # --------------------------------------------------

    def load_data(self):
        datasets = {}

        for dir_path in input_dirs:
            dir_path = Path(dir_path)
            print(f"\n📂 Scanning {dir_path}")

            txt_files = list(dir_path.rglob("*.txt"))
            print(f"Found {len(txt_files)} txt files")

            base_files = [str(p) for p in txt_files if "BASE_DATA_1" in p.name]
            grouper_files = [str(p) for p in txt_files if "GROUPER" in p.name]

            if base_files:
                df_base = dd.concat(
                    [dd.read_csv(fp, sep="\t", dtype=str) for fp in base_files],
                    ignore_index=True
                )
                df_base["TYPE"] = dir_path.name
                datasets[f"base_{dir_path.name}"] = df_base
                print(f"✅ Loaded BASE for {dir_path.name}")

            if grouper_files:
                df_grouper = dd.concat(
                    [dd.read_csv(fp, sep="\t", dtype=str) for fp in grouper_files],
                    ignore_index=True
                )
                df_grouper["TYPE"] = dir_path.name
                datasets[f"grouper_{dir_path.name}"] = df_grouper
                print(f"✅ Loaded GROUPER for {dir_path.name}")

        return datasets

    # --------------------------------------------------
    # STEP 2: GPU Merge
    # --------------------------------------------------

    def merge_data(self, datasets):
        merged_list = []

        for key in datasets:
            if key.startswith("base_"):
                folder = key.replace("base_", "")
                base_df = datasets[key]
                grouper_key = f"grouper_{folder}"

                if grouper_key in datasets:
                    print(f"\n🔗 Merging {folder}")
                    merged = base_df.merge(
                        datasets[grouper_key],
                        on="RECORD_ID",
                        how="inner"
                    )
                    merged_list.append(merged)

        if not merged_list:
            raise ValueError("❌ No datasets merged.")

        final_df = dd.concat(merged_list)
        print("✅ All datasets merged")

        return final_df

    # --------------------------------------------------
    # STEP 3: Train/Test Split on GPU
    # --------------------------------------------------

    def split_and_save(self, df):

        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n✂️ Splitting train/test")

        df = df.persist()

        frac = 0.8
        train = df.sample(frac=frac, random_state=42)
        test = df[~df.index.isin(train.index)]

        train_file = output_dir / "train.csv"
        test_file = output_dir / "test.csv"

        train.repartition(npartitions=1).to_csv(
            train_file,
            index=False,
            single_file=True
        )

        test.repartition(npartitions=1).to_csv(
            test_file,
            index=False,
            single_file=True
        )

        print(f"✅ Train saved → {train_file}")
        print(f"✅ Test saved → {test_file}")

    # --------------------------------------------------

    def close(self):
        self.client.close()
        self.cluster.close()
        print("✅ Cluster closed")


# ==============================
# RUN PIPELINE
# ==============================

def main():
    print("🚀 Starting GPU Pipeline")

    pipeline = GPUPipeline()

    datasets = pipeline.load_data()
    merged_df = pipeline.merge_data(datasets)
    pipeline.split_and_save(merged_df)

    pipeline.close()

    print("\n🎯 FULL GPU PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
