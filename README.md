# GAN-Augmented Startup Success Predictor

Binary classification of startups using WGAN-GP for data augmentation to handle class imbalance.

## Project Description

This project predicts startup success (next funding round, IPO, or acquisition) using a two-stage pipeline:
1. **WGAN-GP** for generating synthetic successful startup data to address class imbalance
2. **MLP Classifier** trained on augmented data for binary classification

### Problem
Only 3-7% of startups are successful, creating severe class imbalance that makes training difficult.

### Solution
Use Wasserstein GAN with Gradient Penalty to generate synthetic minority class samples, improving classifier performance.

### Value
Enables venture capital firms to perform initial screening of startups, reducing due diligence time and improving investment decisions.

## Setup

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Git
- Docker (for deployment)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mipt-2025-mlops
```

2. Create and activate virtual environment:
```bash
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
uv pip install -e ".[dev]"
```

4. Setup pre-commit hooks:
```bash
pre-commit install
```

5. Configure environment variables:
```bash
cp env.example .env
# Edit .env with your Kaggle credentials
```

## Data

### Dataset
- **Source**: [Kaggle - Startup Investments](https://www.kaggle.com/datasets/)
- **Size**: ~50+ MB
- **Records**: ~54,000 companies
- **Class Distribution**: 90-95% unsuccessful vs 5-10% successful

### Data Management
Data is versioned using DVC and split temporally:
- **Train**: Startups before 2015
- **Validation**: 2015-2017
- **Test**: 2018+

## Train

### Download Data
```bash
python -m startup_success_predictor.data.download
```

### Train Models
```bash
python -m startup_success_predictor.train
```

The training pipeline:
1. Trains WGAN-GP on successful startups (minority class)
2. Generates synthetic successful samples
3. Trains MLP classifier on augmented dataset
4. Logs metrics to MLFlow

### Configuration
Modify hyperparameters in `configs/` directory using Hydra.

## Production Preparation

### Export to ONNX
```bash
python -m startup_success_predictor.export_onnx --checkpoint path/to/best.ckpt
```

### Convert to TensorRT (requires GPU)
```bash
bash scripts/convert_trt.sh
```

## Inference

### Local Inference
```bash
python -m startup_success_predictor.infer --input data/test_sample.csv
```

### API Server
```bash
uvicorn startup_success_predictor.app:app --host 0.0.0.0 --port 8000
```

### Docker Deployment
```bash
docker build -t startup-predictor .
docker run -p 8000:8000 startup-predictor
```

## Metrics

- **AUROC**: > 0.80 (target)
- **AUPRC**: > 0.45 (target)
- **F1-Score**: > 0.55 (target)
- **Recall (minority class)**: > 0.70 (target)

## Architecture

```
Data Pipeline: Kaggle → DVC → Polars → PyTorch DataLoader
Training: WGAN-GP → Synthetic Data → MLP Classifier
Deployment: ONNX → FastAPI → Docker
```

## Development

### Code Quality
- Linting & Formatting: `ruff`
- Type Checking: `mypy`
- Pre-commit hooks enforce standards

### Run Tests
```bash
pytest
```

### Run Linters
```bash
ruff check .
ruff format .
mypy startup_success_predictor/
```

## License

MIT

## Author

Хузин Эльдар Русланович (khuzin.er@phystech.edu)


