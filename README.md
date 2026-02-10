# GAN-Augmented Startup Success Predictor

Бинарная классификация стартапов с использованием WGAN-GP для аугментации данных с целью устранения дисбаланса классов.

> Быстрый старт: см. разделы **Установка**, **Обучение** и **Инференс** — они описывают полный путь от клонирования репозитория до получения предсказаний.

## Описание проекта

Этот проект предсказывает успех стартапа (следующий раунд финансирования, IPO или поглощение) с использованием двухэтапного пайплайна:

1.  **WGAN-GP** для генерации синтетических данных успешных стартапов для решения проблемы дисбаланса классов.
2.  **MLP Classifier** (классификатор на основе многослойного перцептрона), обученный на дополненных данных для бинарной классификации.

### Проблема

Только 3-7% стартапов достигают успеха, что создает серьезный дисбаланс классов и затрудняет обучение моделей.

### Решение

Использование Wasserstein GAN with Gradient Penalty (WGAN-GP) для генерации синтетических примеров миноритарного класса, что улучшает производительность классификатора.

### Ценность

Позволяет венчурным фондам проводить первичный отбор стартапов, сокращая время на due diligence и улучшая инвестиционные решения.

## Установка

### Предварительные требования

- Python 3.12+
- Пакетный менеджер [uv](https://github.com/astral-sh/uv)
- Git
- Docker (для развертывания)

### Инструкция по установке

1.  Склонировать репозиторий:

```bash
git clone <repository-url>
cd mipt-2025-mlops
```

2.  Создать и активировать виртуальное окружение:

```bash
uv venv --python 3.12
source .venv/bin/activate  # Для Windows: .venv\Scripts\activate
```

3.  Установить зависимости:

```bash
uv pip install -e ".[dev]"
```

4.  Настроить pre-commit хуки и проверить качество кода:

```bash
pre-commit install
pre-commit run --all-files
```

5.  Настроить переменные окружения для Kaggle (для загрузки данных):

```bash
cp env.example .env
# Отредактировать .env, добавив ваши учетные данные Kaggle (KAGGLE_USERNAME, KAGGLE_KEY)
```

Либо можно использовать стандартный файл `~/.kaggle/kaggle.json` с токеном Kaggle.

## Данные

### Набор данных

- **Источник**: [Kaggle - Big Startup Success/Fail Dataset from Crunchbase](https://www.kaggle.com/datasets/yanmaksi/big-startup-secsees-fail-dataset-from-crunchbase)
- **Размер**: ~50+ МБ
- **Записи**: ~54,000 компаний
- **Распределение классов**: 90-95% неуспешных против 5-10% успешных

### Управление данными

Данные версионируются с использованием DVC и разделены по времени:

- **Train (Обучение)**: Стартапы до 2015 года
- **Validation (Валидация)**: 2015-2017 годы
- **Test (Тест)**: 2018 год и позже

## Обучение

### Загрузка данных

Вариант 1 (рекомендуемый для ДЗ) — через DVC:

```bash
# Однократно: запустить стадию загрузки DVC для наполнения data/raw
dvc repro download
```

Вариант 2 — напрямую через CLI (делает то же самое, что и DVC-стадия):

```bash
python -m startup_success_predictor.cli download-data
```

### Обучение моделей

1. Запустить локальный сервер MLflow для отслеживания экспериментов (по умолчанию пайплайн ожидает сервер на `127.0.0.1:8080`):

```bash
mlflow server --host 127.0.0.1 --port 8080 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

2. Запустить обучение (Hydra + Lightning):

```bash
python -m startup_success_predictor.cli train
```

При необходимости можно переопределить конфиг Hydra, например, уменьшить число эпох или отключить GAN:

```bash
python -m startup_success_predictor.train \
  mlflow.tracking_uri=http://127.0.0.1:8080 \
  train.two_stage.train_gan_first=false \
  train.trainer.max_epochs=1
```

Пайплайн обучения:

1.  Обучает WGAN-GP на успешных стартапах (миноритарный класс).
2.  Генерирует синтетические примеры успешных стартапов.
3.  Обучает MLP классификатор на расширенном наборе данных.
4.  Логирует метрики в MLFlow.

Бейзлайн (без GAN, только MLP классификатор):

```bash
python -m startup_success_predictor.cli train train=baseline
```

Стратифицированная 5-fold оценка (дополнительная проверка стабильности модели):

```bash
python -m startup_success_predictor.eval_kfold \
  train.trainer.max_epochs=5
```

### Конфигурация

Изменять гиперпараметры в директории `configs/` с использованием Hydra.

## Подготовка к продакшену

### Экспорт в ONNX

```bash
python -m startup_success_predictor.cli export-onnx \
  --checkpoint path/to/best.ckpt \
  --input-dim <num_features>
```

### Конвертация в TensorRT (требуется GPU)

```bash
bash scripts/convert_trt.sh
```

## Инференс

### Локальный инференс (из чекпоинта Lightning)

```bash
python -m startup_success_predictor.cli infer \
  --checkpoint path/to/best.ckpt \
  --input-csv data/test_sample.csv \
  --output-csv predictions.csv
```

### API Сервер (ONNX + FastAPI)

1. Убедиться, что у вас есть экспортированная ONNX-модель (`models/classifier.onnx`), см. раздел выше.
2. Запустить сервер:

```bash
uvicorn startup_success_predictor.app:app --host 0.0.0.0 --port 8000
```

3. Пример запроса к `POST /predict`:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {"feature0": 0.1},
    "categorical": {}
  }'
```

### Docker развертывание

Простой запуск только API:

```bash
docker build -t startup-predictor .
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  startup-predictor
```

### Docker Compose (API + MLflow)

```bash
docker-compose up --build
```

По умолчанию поднимаются два сервиса:

- `api` — FastAPI + ONNX модель на порту 8000
- `mlflow` — сервер MLflow на порту 5000

Важно:

- В `docker-compose.yml` используется `./models:/app/models:ro` и `./data:/app/data:ro`, поэтому позаботьтесь, чтобы в `./models` лежала `classifier.onnx`, а в `./data` — нужные CSV.
- Переменные окружения `KAGGLE_USERNAME`, `KAGGLE_KEY`, `MLFLOW_TRACKING_URI` можно пробрасывать через `.env` или вручную при запуске.

## Метрики

- **AUROC**: > 0.80 (цель)
- **AUPRC**: > 0.45 (цель)
- **F1-Score**: > 0.55 (цель)
- **Recall (миноритарный класс)**: > 0.70 (цель)

## Архитектура

```
Пайплайн данных: Kaggle → DVC → Polars → PyTorch DataLoader
Обучение: WGAN-GP → Синтетические данные → MLP Классификатор
Развертывание: ONNX → FastAPI → Docker
```

## Разработка

### Качество кода

- Линтинг и форматирование: `ruff`, `black`, `isort`
- Проверка типов: `mypy`
- Pre-commit хуки обеспечивают соблюдение стандартов (`pre-commit run --all-files`)

### Запуск тестов

```bash
pytest
```

### Запуск линтеров и проверок

```bash
ruff check .
ruff format .
black startup_success_predictor/ tests/
isort startup_success_predictor/ tests/
mypy startup_success_predictor/
pre-commit run --all-files
```

## Лицензия

MIT

## Автор

Хузин Эльдар Русланович (khuzin.er@phystech.edu)
