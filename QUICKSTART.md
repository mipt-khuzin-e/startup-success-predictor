# Quick Start Guide

## Prerequisites

Install the following tools:

```bash
# Install uv (package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Ensure you have Git and Python 3.12+
python --version  # Should be 3.12+
```

## Setup (5 minutes)

```bash
# 1. Navigate to project
cd /Users/khuzin.e/Projects/mipt-2025-mlops

# 2. Create virtual environment
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
uv pip install -e ".[dev]"

# 4. Setup pre-commit hooks
pre-commit install

# 5. Configure environment
cp env.example .env
# Edit .env and add your Kaggle credentials:
#   KAGGLE_USERNAME=your_username
#   KAGGLE_KEY=your_api_key
```

## Download Data

```bash
# Download dataset from Kaggle
python -m startup_success_predictor.data.download

# Version with DVC
dvc add data/raw
git add data/raw.dvc .gitignore
git commit -m "Add dataset"
```

## Training

```bash
# Terminal 1: Start MLFlow server
mlflow server --host 127.0.0.1 --port 8080

# Terminal 2: Train models
python -m startup_success_predictor.train

# View experiments at http://127.0.0.1:8080
```

## Inference

```bash
# Export model to ONNX
python -m startup_success_predictor.export_onnx \
    --checkpoint models/classifier/best.ckpt \
    --input-dim <number_of_features>

# Run inference
python -m startup_success_predictor.infer \
    --checkpoint models/classifier/best.ckpt \
    --input data/test_sample.csv
```

## API Deployment

### Local
```bash
uvicorn startup_success_predictor.app:app --reload
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Docker
```bash
# Build and run
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api

# Stop
docker-compose down
```

## Development

### Code Quality
```bash
# Lint
ruff check startup_success_predictor/

# Format
ruff format startup_success_predictor/

# Type check
mypy startup_success_predictor/

# Run all checks
pre-commit run --all-files
```

### Testing
```bash
pytest  # When tests are added
```

## Project Structure

```
startup_success_predictor/
├── data/
│   ├── download.py      # Download from Kaggle
│   ├── datamodule.py    # PyTorch Lightning DataModule
│   └── preprocessing.py # Polars preprocessing
├── models/
│   ├── gan_module.py        # WGAN-GP
│   ├── classifier_module.py # MLP Classifier
│   └── components/          # Network architectures
├── train.py             # Training pipeline
├── export_onnx.py      # Model export
├── infer.py            # Inference script
└── app.py              # FastAPI service
```

## Configuration

Edit `configs/` to modify:
- `data/startup.yaml` - Data parameters
- `model/gan.yaml` - GAN architecture
- `model/classifier.yaml` - Classifier architecture
- `train/default.yaml` - Training settings

## Troubleshooting

### Import errors
```bash
# Reinstall in editable mode
uv pip install -e .
```

### MLFlow not connecting
```bash
# Check if server is running
curl http://127.0.0.1:8080/health

# Restart server
mlflow server --host 127.0.0.1 --port 8080
```

### DVC errors
```bash
# Reinitialize DVC
dvc init --force
```

## Next Steps

1. ✅ Project setup complete
2. ⏳ Download dataset
3. ⏳ Train models
4. ⏳ Evaluate performance
5. ⏳ Deploy API
6. ⏳ Monitor with MLFlow

## Resources

- [README.md](README.md) - Full documentation
- [IMPLEMENTATION.md](IMPLEMENTATION.md) - Technical details
- [Task PDFs](task/) - Project requirements

