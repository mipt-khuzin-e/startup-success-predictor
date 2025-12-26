# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Common commands

### Environment & installation

- Create virtualenv (Python 3.12) and activate (macOS/Linux):
  ```bash
  uv venv --python 3.12
  source .venv/bin/activate
  ```
- Install project with dev dependencies:
  ```bash
  uv pip install -e ".[dev]"
  ```
- Install pre-commit hooks:
  ```bash
  pre-commit install
  ```
- Bootstrap local env vars from example file:
  ```bash
  cp env.example .env
  # Fill in Kaggle and (optionally) MLflow settings
  ```

### Data

- Download startup dataset from Kaggle (uses credentials from `.env`):
  ```bash
  python -m startup_success_predictor.data.download
  ```
  This populates `data/raw` and validates that CSVs were downloaded.

### Training

- Default two-stage training (WGAN-GP + MLP classifier via Hydra):
  ```bash
  python -m startup_success_predictor.train
  ```
- Override Hydra config values at the CLI, e.g. different learning rate or number of GAN samples:
  ```bash
  python -m startup_success_predictor.train \
    model.training.learning_rate=0.0005 \
    model.generation.n_samples=10000
  ```

### Evaluation, export, inference

- Export best classifier checkpoint to ONNX:
  ```bash
  python -m startup_success_predictor.export_onnx \
    --checkpoint path/to/best.ckpt \
    --input-dim <num_features>
  ```
- Local batch inference from a CSV with a Lightning checkpoint:
  ```bash
  python -m startup_success_predictor.infer \
    --checkpoint path/to/best.ckpt \
    --input data/test_sample.csv \
    --output predictions.csv
  ```
- Run FastAPI server backed by an exported ONNX model (expects `models/classifier.onnx`):
  ```bash
  uvicorn startup_success_predictor.app:app --host 0.0.0.0 --port 8000
  ```

### TensorRT conversion

- Convert ONNX model to TensorRT engine (requires `trtexec` from TensorRT):
  ```bash
  bash scripts/convert_trt.sh \
    --onnx models/classifier.onnx \
    --output models/classifier.engine
  ```

### Quality checks & tests

- Run full test suite (pytest):
  ```bash
  pytest
  ```
- Run a narrower subset of tests (single file or test expression):
  ```bash
  pytest path/to/test_file.py -k "test_name_fragment"
  ```
- Linting and formatting with Ruff:
  ```bash
  ruff check .
  ruff format .
  ```
- Type checking (strict mypy settings):
  ```bash
  mypy startup_success_predictor/
  ```

## High-level architecture

### Top-level layout

- `pyproject.toml` — defines the `startup-success-predictor` package, runtime and dev dependencies, and tooling configuration for Ruff and mypy.
- `startup_success_predictor/` — main Python package containing configuration, data pipeline, models, training, export, inference, and API server code.
- `configs/` — Hydra configuration tree for data, model, and training; this is the primary way to control experiments from the CLI.
- `scripts/convert_trt.sh` — shell wrapper for converting ONNX models to TensorRT engines.
- `env.example` — template for `.env` consumed by Pydantic `Settings`.

The project is organized as a two-stage ML pipeline (WGAN-GP → classifier) with clear separation between configuration, data handling, model logic, and serving.

### Configuration and settings

- **Hydra configs (`configs/`)**
  - `configs/config.yaml` is the root Hydra config; it composes:
    - `data: startup` → `configs/data/startup.yaml` (file name, batch size, temporal split dates, categorical columns, etc.).
    - `model: classifier` or `model: gan` → `configs/model/*.yaml` (MLP vs WGAN-GP hyperparameters and metrics).
    - `train: default` → `configs/train/default.yaml` (trainer settings, early stopping, checkpointing, and two-stage training flags).
  - `configs/config.yaml` also defines global paths (`paths.*`), MLflow experiment metadata, and logging directories that are used by the training script.
  - Training entrypoint `startup_success_predictor/train.py` is annotated with `@hydra.main(config_path="../configs", config_name="config")`, so Hydra will:
    - Use `configs/` as the config tree.
    - Change the working directory to a Hydra run directory. File-system paths should therefore come from `cfg.paths.*` or Pydantic settings rather than assuming `cwd`.

- **Runtime settings (`startup_success_predictor/config.py`)**
  - Uses `pydantic_settings.BaseSettings` to load environment-driven configuration for:
    - Kaggle credentials (`kaggle_username`, `kaggle_key`) — mapped from `KAGGLE_USERNAME`, `KAGGLE_KEY` in `.env`.
    - MLflow tracking URI default (`mlflow_tracking_uri`) — for components that want to read it from env instead of Hydra.
    - Project-relative paths: `data_dir`, `raw_data_dir`, `processed_data_dir`, `models_dir`, `plots_dir` rooted at the package directory.
  - `get_settings()` is the single factory used across modules to create a `Settings` instance.

Hydra is responsible for **experiment configuration and logging**, while Pydantic settings are responsible for **environment-dependent IO and credentials** (Kaggle, data/ model locations) shared across scripts.

### Data ingestion and preprocessing

- **Data download (`startup_success_predictor.data.download`)**
  - Reads Kaggle credentials from `Settings` and exports them as `KAGGLE_USERNAME`/`KAGGLE_KEY` for the official Kaggle API client.
  - Downloads a Crunchbase-based startup dataset from Kaggle into `settings.raw_data_dir` and unzips it.
  - Provides a `validate_data` routine that lists CSV files and basic shapes; the script’s CLI flow recommends the next DVC steps (adding the raw data to DVC and committing the corresponding `.dvc` files).

- **Preprocessing utilities (`startup_success_predictor.data.preprocessing`)**
  - `handle_missing_values` — centralizes missing-value policy (drop/mean/median/fill) using Polars.
  - `encode_categorical` — supports one-hot and label encoding; in practice, the `StartupDataModule` uses label encoding and stores per-column category → index mappings.
  - `normalize_features` — z-score normalization with statistics computed on the training split and re-used for validation and test.
  - `temporal_split` — splits the full dataframe into train/val/test based on date thresholds (`train_end`, `val_end`) defined in Hydra config.
  - `polars_to_tensor` — bridges Polars DataFrames to PyTorch tensors and is used consistently throughout the data module.

- **Data module (`startup_success_predictor.data.datamodule.StartupDataModule`)**
  - Entry point for data in the Lightning training loop.
  - Responsibilities:
    - Load CSV via Polars using `data_path` determined from `Settings.raw_data_dir` and `cfg.data.data_file`.
    - Convert the raw `status` field into a binary `target` column (`1` for acquired/IPO/operating, `0` for closed), enforcing the project’s definition of success.
    - Perform temporal splitting into train/val/test using the configured dates.
    - Identify numerical vs categorical features from the full column set, excluding the target and date columns; categorical columns are driven by Hydra config.
    - Apply label encoding to categorical columns and z-score normalization to numerical columns, fitting only on training data and reusing parameters for val/test.
    - Build `TensorDataset`s and `DataLoader`s for all three splits, with persistent workers when `num_workers > 0`.
    - Provide `get_minority_class_data()` to retrieve only minority-class (successful) samples from the training set for GAN training.

The **data path** is therefore:
Kaggle CSV → `StartupDataModule` (Polars + preprocessing) → PyTorch `TensorDataset` → Lightning `DataLoader`s.

### Modeling and training pipeline

- **GAN components (`startup_success_predictor.models.components.gan`)**
  - `Generator` — MLP that maps latent vectors to feature space with optional batch norm and final `tanh` activation.
  - `Critic` — MLP that scores samples for Wasserstein distance, with LeakyReLU and dropout, no final activation.
  - `compute_gradient_penalty` — implements the WGAN-GP gradient penalty between real and interpolated samples.

- **GAN Lightning module (`startup_success_predictor.models.gan_module.WGANGPModule`)**
  - Wraps `Generator` and `Critic` into a manual-optimization LightningModule with WGAN-GP training.
  - Hyperparameters (latent dim, hidden sizes, learning rates, `n_critic`, `lambda_gp`) are driven entirely by `configs/model/gan.yaml`.
  - Implements `generate_samples(n_samples)` for downstream use; this is how the classifier training phase gets synthetic minority-class data.

- **Classifier components (`startup_success_predictor.models.components.classifier` and `.classifier_module`)**
  - `MLPClassifier` is a plain nn.Module that builds a LeakyReLU MLP with optional batch norm and dropout.
  - `ClassifierModule` is the LightningModule used in training and inference:
    - Configured from `configs/model/classifier.yaml` (architecture, optimizer, `pos_weight` for class imbalance, and tracked metrics).
    - Uses `BCEWithLogitsLoss` with configurable positive-class weight.
    - Tracks AUROC, AUPRC, F1, precision, recall, and accuracy for train/val/test via TorchMetrics.

- **End-to-end training orchestration (`startup_success_predictor.train`)**
  - `main(cfg)` wires everything together under Hydra control:
    - Sets RNG seed from `cfg.seed`.
    - Builds an `MLFlowLogger` using MLflow settings from the Hydra config (not the Pydantic settings).
    - Logs the current git commit hash into MLflow via `get_git_commit_id()` and logs the entire resolved config as hyperparameters.
    - Resolves the data CSV path from `Settings.raw_data_dir` plus `cfg.data.data_file`, falling back to the first CSV in that directory if the configured file is missing.
    - Constructs and sets up `StartupDataModule` with the configured splits and categorical columns.
  - **Two-stage training logic** (controlled by `configs/train/default.yaml`):
    1. **GAN phase** (if `two_stage.train_gan_first`): trains `WGANGPModule` on minority-class-only data and saves the best critic/generator checkpoint under `cfg.paths.models_dir/gan`.
    2. **Synthetic data generation** (if `two_stage.use_synthetic_data`): uses `generate_samples` to create synthetic positive samples.
    3. **Classifier phase**: augments the real training dataset with synthetic positives (when available), constructs an augmented `TensorDataset`, and trains `ClassifierModule` with Lightning `Trainer`, early stopping, and model checkpointing parameters from `cfg.train`.
  - After training, the script runs a test pass on the test split and prints a summary, while MLflow captures logged metrics.

This design centralizes **experiment logic** in a single training script while keeping model internals and data preprocessing modular and reusable.

### Export, inference, and serving

- **Checkpoint → ONNX (`startup_success_predictor.export_onnx`)**
  - Loads a trained `ClassifierModule` from a Lightning checkpoint via `.load_from_checkpoint`.
  - Exports to ONNX with a dynamic batch dimension and standard `input`/`output` node names.
  - Validates the resulting ONNX model by loading and running `onnx.checker.check_model`.
  - Default output path is `Settings.models_dir / "classifier.onnx"` if `--output` is not provided.

- **Offline inference (`startup_success_predictor.infer`)**
  - Loads a Lightning checkpoint and a CSV into a Polars DataFrame.
  - Current implementation assumes input is already preprocessed and uses a simplified path: converts raw numeric columns directly to a tensor, runs the classifier, and appends `prediction` and `probability` columns before writing a new CSV.
  - The script prints a summary of positive vs negative predictions; in a productionized pipeline, this would be extended to load and apply stored preprocessing metadata (`feature_cols`, encoders, normalization stats).

- **FastAPI serving (`startup_success_predictor.app`)**
  - Defines a FastAPI app exposing:
    - `GET /` — basic status/info.
    - `GET /health` — indicates whether the ONNX model has been loaded.
    - `POST /predict` and `POST /predict_batch` — accept numeric and categorical features, run the ONNX model, apply a sigmoid to logits, and return success flag plus probability.
  - On startup:
    - Reads `Settings.models_dir` and looks for `classifier.onnx`.
    - If present, creates a global `onnxruntime.InferenceSession` on CPU.
    - If missing, the app starts but prediction endpoints return 503 until a model is available.

- **Deployment**
  - README documents a basic Docker flow:
    - Build: `docker build -t startup-predictor .`
    - Run: `docker run -p 8000:8000 startup-predictor`
  - The container is expected to run `uvicorn` against `startup_success_predictor.app:app`, serving the ONNX-backed API.

This end-to-end flow is: **Kaggle data → Lightning GAN + classifier training (Hydra + MLflow) → checkpoint → ONNX export → FastAPI/ Docker/ TensorRT deployment paths**.
