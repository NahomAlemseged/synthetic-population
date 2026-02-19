import argparse
import pandas as pd
import numpy as np
import torch
import yaml
import os
import time
from pathlib import Path
from ctgan import CTGAN

# --------------------------
# Command line arguments
# --------------------------
parser = argparse.ArgumentParser(description="Synthetic population generation with CTGAN")
parser.add_argument("--n_samples", type=int, required=True)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--sample_rows", type=int, default=None)
parser.add_argument("--num_processes", type=int, default=None, help="Number of CPU threads for parallelism")  # <- ADD THIS
args = parser.parse_args()

num_processes = args.num_processes if args.num_processes else os.cpu_count()

n_samples = args.n_samples
epochs = args.epochs
sample_rows = args.sample_rows

# --------------------------
# A100 GPU Setup
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)   # DO NOT use all cores on Colab
else:
    torch.set_num_threads(os.cpu_count())

# --------------------------
# Load YAML config
# --------------------------
CONFIG_PATH = Path("/content/synthetic-population_/config/params.yaml")
with open(CONFIG_PATH, "r") as f:
    params_ = yaml.safe_load(f)

INPUT_CSV = Path(params_["generate"]["input"])
OUTPUT_PATH = Path(params_["generate"]["output"])
OUTPUT_CSV = OUTPUT_PATH / "synthetic_emergency.csv"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# --------------------------
# Synthetic Generator
# --------------------------
class SyntheticGenerator:

    def generate_ipf(self, df, features, target_marginals, tol=1e-5, max_iter=100):

        df = df.copy()
        df["weight"] = 1.0

        for iteration in range(max_iter):
            old_weights = df["weight"].copy()

            for feat in features:
                current = df.groupby(feat)["weight"].sum()
                desired = pd.Series(target_marginals[feat])
                ratios = desired / current
                df["weight"] *= df[feat].map(ratios)

            if np.allclose(df["weight"], old_weights, atol=tol):
                print(f"✅ IPF converged at iteration {iteration}")
                break

        synthetic_demographics = df.sample(
            n=min(n_samples, len(df)),
            weights="weight",
            replace=True,
            random_state=42
        ).drop(columns=["weight"])

        if "APR_MDC" in synthetic_demographics.columns:
            synthetic_demographics = synthetic_demographics.drop(columns=["APR_MDC"])

        return synthetic_demographics

    # --------------------------
    # GPU-Optimized CTGAN
    # --------------------------
    def learn_ctgan(self, df_real, features, target_col="APR_MDC", epochs=10):

        columns_to_use = features + [target_col]
        df_ctgan = df_real[columns_to_use].copy()

        for col in columns_to_use:
            df_ctgan[col] = df_ctgan[col].astype("category")

        print(f"🔥 Training CTGAN on {device}")

        # A100 can handle much larger batches
        batch_size = 1024
        pac = 10
        batch_size -= batch_size % pac

        ctgan = CTGAN(
            epochs=epochs,
            batch_size=batch_size,
            embedding_dim=128,
            generator_dim=(256, 256),
            discriminator_dim=(256, 256),
            verbose=True,
            cuda=(device.type == "cuda")
        )

        ctgan.fit(df_ctgan, discrete_columns=columns_to_use)

        print("✅ CTGAN training complete")
        return ctgan

    def generate_gan(self, ctgan, synthetic_demographics, target_col="APR_MDC"):

        df_demo = synthetic_demographics.copy()
        for col in df_demo.columns:
            df_demo[col] = df_demo[col].astype("category")

        synthetic_target = ctgan.sample(len(df_demo))

        if target_col in df_demo.columns:
            df_demo = df_demo.drop(columns=[target_col])

        df_final = pd.concat(
            [df_demo.reset_index(drop=True),
             synthetic_target[[target_col]].reset_index(drop=True)],
            axis=1
        )

        print(f"✅ Final synthetic dataset shape: {df_final.shape}")
        return df_final


# --------------------------
# Main
# --------------------------
def main():
    print("⚙️ Starting A100-optimized synthetic pipeline")
    start_time = time.time()

    df_real = pd.read_csv(INPUT_CSV, dtype=str)
    print(f"📂 Loaded {len(df_real):,} rows")

    if sample_rows:
        df_real = df_real.sample(sample_rows, random_state=42)
        print(f"⚡ Using subset of {len(df_real):,} rows")

    features = [
        "SEX_CODE", "PAT_AGE", "RACE", "ETHNICITY",
        "PAT_ZIP", "PAT_COUNTY", "PUBLIC_HEALTH_REGION"
    ]

    target_col = "APR_MDC"

    target_marginals = {
        col: df_real[col].value_counts().to_dict()
        for col in features
    }

    synth = SyntheticGenerator()

    print("🔹 Step 1: IPF")
    synthetic_demographics = synth.generate_ipf(
        df_real, features, target_marginals
    )

    print("🔹 Step 2: CTGAN (GPU)")
    ctgan_model = synth.learn_ctgan(
        df_real, features, target_col, epochs
    )

    print("🔹 Step 3: Generate APR_MDC")
    synthetic_dataset = synth.generate_gan(
        ctgan_model, synthetic_demographics
    )

    synthetic_dataset.to_csv(OUTPUT_CSV, index=False)

    end_time = time.time()
    print(f"⏱ Total time: {end_time - start_time:.2f} sec")
    print(f"💾 Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

