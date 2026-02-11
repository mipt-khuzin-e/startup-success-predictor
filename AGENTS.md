# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Common commands

### Environment & installation

Create virtualenv (Python 3.12+) and activate:

```bash
uv venv --python 3.12
source .venv/bin/activate
```

Install project with dev dependencies:

```bash
uv pip install -e ".[dev]"
```

Install pre-commit hooks:

```bash
pre-commit install
```

Bootstrap local env vars (Kaggle credentials for data download, optional MLflow):

```bash
cp env.example .env
# Edit .env to add KAGGLE_USERNAME and KAGGLE_KEY
```

### Data management

Download Startup Investments dataset from Kaggle (recommended via DVC):

```bash
dvc repro download
```

Or directly via CLI:

```bash
python -m startup_success_predictor.cli download-data
```

### Training

Default two-stage training (WGAN-GP + MLP classifier):

```bash
python -m startup_success_predictor.cli train
```

Override Hydra config at CLI (e.g., learning rate, GAN samples):

```bash
python -m startup_success_predictor.train \
  model.training.learning_rate=0.0005 \
  model.generation.n_samples=10000
```

Baseline training (classifier only, no GAN augmentation):

```bash
python -m startup_success_predictor.cli train train=baseline
```

Stratified 5-fold cross-validation:

```bash
python -m startup_success_predictor.eval_kfold \
  train.trainer.max_epochs=5
```

Optional: Start MLflow UI to view experiments:

```bash
mlflow server --host 127.0.0.1 --port 8080 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

### Export and inference

Export checkpoint to ONNX:

```bash
python -m startup_success_predictor.cli export-onnx \
  --checkpoint path/to/best.ckpt \
  --input-dim <num_features>
```

Batch inference from checkpoint:

```bash
python -m startup_success_predictor.cli infer \
  --checkpoint path/to/best.ckpt \
  --input-csv data/test_sample.csv \
  --output-csv predictions.csv
```

Run FastAPI server (requires `models/classifier.onnx`):

```bash
uvicorn startup_success_predictor.app:app --host 0.0.0.0 --port 8000
```

### Testing and quality

Run full test suite:

```bash
pytest
```

Run specific tests:

```bash
pytest path/to/test_file.py -k "test_name_fragment"
```

Linting and formatting:

```bash
ruff check .
ruff format .
```

Type checking (strict mypy):

```bash
mypy startup_success_predictor/
```

Run all pre-commit hooks:

```bash
pre-commit run --all-files
```

### Docker deployment

Build and run API container:

```bash
docker build -t startup-predictor .
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  startup-predictor
```

Docker Compose (API + MLflow):

```bash
docker-compose up --build
```

## High-level architecture

### Pipeline overview

This is a two-stage ML pipeline for binary classification of startup success:

1. **WGAN-GP** generates synthetic successful startups (minority class) to address class imbalance
2. **MLP Classifier** trains on augmented data (real + synthetic)

Data flow: `Kaggle CSV → StartupDataModule (Polars + preprocessing) → PyTorch DataLoader → Lightning training → ONNX → FastAPI/Docker deployment`

### Configuration system

**Hydra configs** (`configs/`):

- `configs/config.yaml` — root config composing data, model, and train configs
- `configs/data/startup.yaml` — data file, batch size, temporal splits, categorical columns
- `configs/model/*.yaml` — hyperparameters for classifier and GAN
- `configs/train/*.yaml` — trainer settings, early stopping, two-stage flags

**Pydantic Settings** (`startup_success_predictor/config.py`):

- Loads environment vars from `.env` (Kaggle credentials, MLflow URI)
- Defines project paths: `data_dir`, `raw_data_dir`, `processed_data_dir`, `models_dir`, `plots_dir`
- Access via `get_settings()` singleton

**Critical**: Hydra changes working directory to a run-specific folder. Always use `cfg.paths.*` or Pydantic settings for file paths, never assume `cwd`.

### Data module

`StartupDataModule` (`startup_success_predictor.data.datamodule`):

- Loads CSV via Polars
- Converts `status` field to binary `target` (1=success: acquired/IPO/operating, 0=closed)
- Temporal split: train (pre-2015), val (2015-2017), test (2018+) based on Hydra config
- Label encodes categorical features, z-score normalizes numerical features (fit on train only)
- Provides `get_minority_class_data()` for GAN training on successful startups

Preprocessing utilities in `startup_success_predictor.data.preprocessing`:

- `handle_missing_values`, `encode_categorical`, `normalize_features`, `temporal_split`, `polars_to_tensor`

### Models

**GAN** (`startup_success_predictor.models.gan_module.WGANGPModule`):

- `Generator`: MLP mapping latent vectors to feature space
- `Critic`: MLP scoring samples for Wasserstein distance
- Implements WGAN-GP with gradient penalty
- `generate_samples(n_samples)` creates synthetic minority-class data

**Classifier** (`startup_success_predictor.models.classifier_module.ClassifierModule`):

- `MLPClassifier`: LeakyReLU MLP with batch norm and dropout
- Uses `BCEWithLogitsLoss` with configurable `pos_weight` for class imbalance
- Tracks AUROC, AUPRC, F1, precision, recall, accuracy via TorchMetrics

### Training orchestration

`startup_success_predictor.train.main()`:

1. Sets RNG seed, builds MLFlowLogger, logs git commit and config
2. Constructs `StartupDataModule` with temporal splits
3. **Two-stage training** (if `two_stage.train_gan_first=true`):
   - Train WGAN-GP on minority class only
   - Generate synthetic positive samples
   - Augment training data with synthetics
   - Train classifier on augmented dataset
4. Test on hold-out test set, log metrics to MLflow

### Serving

**ONNX Export** (`startup_success_predictor.export_onnx`):

- Loads Lightning checkpoint, exports to ONNX with dynamic batch dimension
- Default output: `models/classifier.onnx`

**FastAPI** (`startup_success_predictor.app`):

- `GET /` — status
- `GET /health` — model load status
- `POST /predict`, `POST /predict_batch` — inference via ONNX Runtime
- Loads `models/classifier.onnx` on startup

**CLI** (`startup_success_predictor.cli`):

- Typer-based CLI wrapping train, export-onnx, infer, download-data commands
- Use `python -m startup_success_predictor.cli <command>` for all operations

### Package structure

```
startup_success_predictor/
├── cli.py              # Typer CLI entrypoint
├── config.py           # Pydantic settings
├── train.py            # Main training orchestration
├── export_onnx.py      # Checkpoint → ONNX conversion
├── infer.py            # Batch inference utilities
├── app.py              # FastAPI server
├── eval_kfold.py       # K-fold cross-validation
├── data/
│   ├── download.py     # Kaggle Startup Investments download
│   ├── preprocessing.py # Polars-based preprocessing
│   └── datamodule.py   # PyTorch Lightning DataModule
├── models/
│   ├── components/
│   │   ├── gan.py      # Generator, Critic, gradient penalty
│   │   └── classifier.py # MLPClassifier nn.Module
│   ├── gan_module.py   # WGANGPModule (LightningModule)
│   └── classifier_module.py # ClassifierModule (LightningModule)
└── utils/
    └── metrics.py      # Custom metrics
```

### Key design patterns

1. **Temporal splitting**: Train/val/test split by date to prevent data leakage
2. **Class imbalance handling**: WGAN-GP generates synthetic minority class + `pos_weight` in loss
3. **Modular preprocessing**: Fit encoders/normalizers on train split, reuse on val/test
4. **Hydra composition**: Swap configs (`model: gan` vs `model: classifier`) without code changes
5. **MLflow tracking**: All experiments logged with hyperparameters, metrics, and git commit
6. **ONNX deployment**: Export to ONNX for production serving
