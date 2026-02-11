# GAN-Augmented Startup Success Predictor

Хузин Эльдар Русланович (khuzin.er@phystech.edu)

Бинарная классификация стартапов с использованием WGAN-GP для аугментации данных с целью устранения дисбаланса классов.

## Постановка задачи

- **Сухой остаток:** бинарная классификация стартапов на успешные (получившие следующий раунд финансирования, IPO или acquisition) и неуспешные (закрытые или не получившие финансирования) — предсказание данной характеристики.
- **Решаемая проблема:** дисбаланс классов в данных (только 3-7% стартапов успешны) — на таких данных сложно обучать модели.
- **Основа решения:** обучение и применение Wasserstein GAN для генерации синтетических данных.
- **Ценность:** система может использоваться венчурными фондами для первичного скрининга стартапов, снижая время на due diligence и улучшая качество инвестиционных решений.

## Формат входных и выходных данных

- **Input:** вектор признаков стартапа — POST-запрос в формате:

  ```json
  {
    "features": { "feature_0": 0.1, "feature_1": 0.5 },
    "categorical": { "category_0": 1 }
  }
  ```

  Числовые и категориальные признаки передаются отдельно.

- **Output:** ответ в формате:

  ```json
  { "success": true, "probability": 0.87 }
  ```

- **Протокол:** HTTP, FastAPI-сервер, обрабатывающий POST-запрос на `/predict`.

## Датасет

- **Источник:** [Kaggle — Startup Success/Fail Dataset from Crunchbase](https://www.kaggle.com/datasets/yanmaksi/big-startup-secsees-fail-dataset-from-crunchbase)
- **Размер:** ~50+ MB (CSV-файл), ~54 000 компаний
- **Дисбаланс:** соотношение примерно 90-95% негативных vs 5-10% позитивных примеров — идеальный сценарий для GAN-аугментации

Известные сложности: пропущенные значения, необходимость feature engineering, устаревшие данные, влияние временного периода на успешность.

## Метрики

**Основные:**

| Метрика                     | Целевое значение | Описание                                            |
| --------------------------- | ---------------- | --------------------------------------------------- |
| AUROC                       | > 0.80           | Основная метрика, инвариантная к threshold          |
| AUPRC                       | > 0.45           | Информативна при сильном дисбалансе (baseline ≈ 5%) |
| F1-score                    | > 0.55           | Гармоническое среднее precision и recall            |
| Recall (миноритарный класс) | > 0.70           | Важно не пропустить успешные стартапы               |

**Дополнительные:** Precision@k (k=100, 500) — имитирует реальный сценарий VC.

## Валидация и тест

- **Temporal Split:**
  - Train: стартапы до 2015 года
  - Validation: 2015–2017
  - Test: 2018+
- **Stratified K-Fold:** 5 фолдов с сохранением пропорции классов для оценки стабильности
- **Воспроизводимость:** фиксированный `random_seed=30`, версионирование через DVC, логирование в MLflow, фиксация зависимостей через `uv.lock`

> **Замечание:** GAN обучается _только_ на train set. Синтетические данные генерируются для train set и не используются в validation/test для честной оценки.

## Моделирование

### Основная модель

Двухэтапный pipeline:

1. **Этап 1 — WGAN-GP** для генерации синтетических данных миноритарного класса:
   - Generator: `Linear(100 → 256) → BatchNorm → LeakyReLU → Linear(256 → 128) → BatchNorm → LeakyReLU → Linear(128 → num_features) → Tanh`
   - Critic: `Linear(num_features → 128) → LeakyReLU → Dropout(0.3) → Linear(128 → 64) → LeakyReLU → Linear(64 → 1)`
   - Loss: Wasserstein loss + Gradient Penalty

2. **Этап 2 — MLP Classifier** на расширенном наборе данных:
   - Архитектура: `MLP [num_features → 256 → 128 → 64 → 1]` с BatchNorm, LeakyReLU, Dropout
   - Loss: Binary Cross-Entropy
   - Optimizer: Adam с регуляризацией

---

## Setup

### Предварительные требования

- [Python 3.12+](https://www.python.org/downloads/)
- Пакетный менеджер [uv](https://github.com/astral-sh/uv)
- [Git](https://git-scm.com/)
- [Docker](https://www.docker.com/) (для развертывания)

### Установка

1. Склонировать репозиторий:

```bash
git clone git@github.com:mipt-khuzin-e/startup-success-predictor.git
cd startup-success-predictor
```

2. Создать окружение и установить зависимости:

```bash
uv sync
```

3. Настроить pre-commit хуки:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

4. Настроить переменные окружения для Kaggle (для загрузки данных):

```bash
cp env.example .env
# Отредактировать .env, добавив ваши учетные данные Kaggle (KAGGLE_USERNAME, KAGGLE_KEY)
```

Либо можно использовать стандартный файл `~/.kaggle/kaggle.json` с токеном Kaggle.

## Train

### 1. Загрузка данных

Вариант 1 (рекомендуемый) — через DVC:

```bash
uv run dvc repro download
```

Вариант 2 — напрямую через CLI:

```bash
uv run python -m startup_success_predictor.cli download-data
```

### 2. Обучение моделей

Запустить локальный сервер MLflow для отслеживания экспериментов:

```bash
uv run mlflow server --host 127.0.0.1 --port 8080 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

> **Примечание:** Для локального обучения используется порт 8080. При развертывании через Docker Compose MLflow будет доступен на порту 5000.

Запустить обучение основной модели (WGAN-GP + MLP, Hydra + Lightning):

```bash
uv run python -m startup_success_predictor.cli train
```

При необходимости можно переопределить параметры Hydra:

```bash
uv run python -m startup_success_predictor.train \
  mlflow.tracking_uri=http://127.0.0.1:8080 \
  train.two_stage.train_gan_first=false \
  train.trainer.max_epochs=1
```

### 3. Дополнительная оценка

Стратифицированная 5-fold оценка стабильности модели:

```bash
uv run python -m startup_success_predictor.eval_kfold \
  train.trainer.max_epochs=5
```

### Пайплайн обучения

1. Обучает WGAN-GP на успешных стартапах (миноритарный класс).
2. Генерирует синтетические примеры успешных стартапов.
3. Обучает MLP-классификатор на расширенном наборе данных.
4. Логирует метрики и артефакты в MLflow.

### Конфигурация

Гиперпараметры настраиваются в директории `configs/`:

- `configs/config.yaml` — главный конфиг (seed, пути, MLflow)
- `configs/model/classifier.yaml` — архитектура MLP (hidden_dims, dropout, pos_weight)
- `configs/model/gan.yaml` — параметры WGAN-GP
- `configs/train/default.yaml` — параметры обучения (epochs, early stopping, checkpointing)
- `configs/train/baseline.yaml` — конфиг бейзлайна
- `configs/data/startup.yaml` — настройки данных

## Production preparation

### Экспорт в ONNX

Конвертация обученного Lightning-чекпоинта в ONNX-формат для инференса:

```bash
uv run python -m startup_success_predictor.cli export-onnx \
  --checkpoint "path/to/best.ckpt" \
  --output models/classifier.onnx
```

Параметры `--checkpoint`, `--output` и `--input-dim` определяются автоматически и могут быть опущены:

```bash
uv run python -m startup_success_predictor.cli export-onnx
```

### Артефакты для поставки

Для запуска модели в продакшене необходимы:

| Артефакт     | Путь                               | Описание                               |
| ------------ | ---------------------------------- | -------------------------------------- |
| ONNX-модель  | `models/classifier.onnx`           | Экспортированная модель классификатора |
| Приложение   | `startup_success_predictor/app.py` | FastAPI-сервер для инференса           |
| Конфигурация | `configs/`                         | Параметры модели                       |

Зависимости инференса минимальны: `fastapi`, `uvicorn`, `onnxruntime`, `numpy` — код предсказания (`app.py`) отделен от обучения (`train.py`) и не требует `torch`, `lightning` и других тяжелых пакетов.

## Infer

### Локальный инференс (из чекпоинта Lightning)

```bash
uv run python -m startup_success_predictor.cli infer \
  --checkpoint path/to/best.ckpt \
  --input-csv data/test_sample.csv \
  --output-csv predictions.csv
```

**Формат входного CSV:** файл должен содержать столбцы с признаками стартапа (числовые и категориальные). Пример данных можно получить из `data/raw` после загрузки датасета.

**Формат выходного CSV:** файл `predictions.csv` с колонками оригинальных данных и добавленными предсказаниями.

### API-сервер (ONNX + FastAPI)

1. Убедиться, что ONNX-модель экспортирована (см. раздел Production preparation).

2. Запустить сервер:

```bash
uv run uvicorn startup_success_predictor.app:app --host 0.0.0.0 --port 8000
```

3. Эндпоинты:

- `GET /` — статус сервера
- `GET /health` — проверка здоровья (статус модели)
- `POST /predict` — предсказание для одного стартапа
- `POST /predict_batch` — пакетное предсказание

4. Примеры запросов:

**Одиночное предсказание** (`POST /predict`):

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {"feature_1": 0.5, "feature_2": 1.2},
    "categorical": {"category_1": 0, "category_2": 1}
  }'
```

Ответ:

```json
{ "success": true, "probability": 0.87 }
```

**Пакетное предсказание** (`POST /predict_batch`):

```bash
curl -X POST "http://localhost:8000/predict_batch" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "features": {"feature_1": 0.5, "feature_2": 1.2},
      "categorical": {"category_1": 0, "category_2": 1}
    },
    {
      "features": {"feature_1": 0.1, "feature_2": 0.3},
      "categorical": {"category_1": 2, "category_2": 0}
    }
  ]'
```

Ответ:

```json
[
  { "success": true, "probability": 0.87 },
  { "success": false, "probability": 0.12 }
]
```

### Docker-развертывание

Запуск только API:

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

Сервисы:

- `api` — FastAPI + ONNX-модель на порту **8000**
- `mlflow` — сервер MLflow на порту **5000**

Для работы необходимо:

- Экспортировать обученную модель в ONNX (см. раздел "Экспорт в ONNX") и поместить `classifier.onnx` в `./models/`
- Переменные окружения (`KAGGLE_USERNAME`, `KAGGLE_KEY`, `MLFLOW_TRACKING_URI`) задаются через `.env` или при запуске

> **Примечание:** При использовании Docker Compose MLflow будет доступен на `http://localhost:5000`, API на `http://localhost:8000`.

## Разработка

### Качество кода

- Линтинг и форматирование: `ruff`
- Проверка типов: `mypy`
- Pre-commit хуки обеспечивают соблюдение стандартов

### Запуск тестов

```bash
uv run pytest
```

### Запуск линтеров

```bash
uv run ruff check .
uv run ruff format .
uv run mypy startup_success_predictor/
```

### Pre-commit хуки

```bash
# Установить хуки (если еще не установлены)
uv run pre-commit install

# Запустить на всех файлах
uv run pre-commit run --all-files
```

## Лицензия

MIT
