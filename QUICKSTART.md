# Руководство по быстрому старту (Quick Start)

## Предварительные требования

Установить следующие инструменты:

```bash
# Установка uv (пакетный менеджер)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Убедиться, что у вас установлены Git и Python 3.12+
python --version  # Должно быть 3.12+
```

## Настройка (5 минут)

```bash
# 1. Перейти в директорию проекта
cd /Users/khuzin.e/Projects/mipt-2025-mlops

# 2. Создать виртуальное окружение
uv venv --python 3.12
source .venv/bin/activate  # Для Windows: .venv\Scripts\activate

# 3. Установить зависимости
uv pip install -e ".[dev]"

# 4. Настроить pre-commit хуки
pre-commit install

# 5. Настроить окружение
cp env.example .env
# Отредактировать .env и добавить ваши учетные данные Kaggle:
#   KAGGLE_USERNAME=ваше_имя_пользователя
#   KAGGLE_KEY=ваша_api_ключ
```

## Загрузка данных (через стадию DVC)

```bash
# Заполнение data/raw с использованием стадии загрузки под управлением DVC
dvc repro download
```

## Обучение

```bash
# Терминал 1: Запуск сервера MLFlow
mlflow server --host 127.0.0.1 --port 8080

# Терминал 2: Обучение моделей через Typer CLI
python -m startup_success_predictor.cli train

# Просмотр экспериментов по адресу http://127.0.0.1:8080
```

## Инференс

```bash
# Экспорт модели в ONNX
python -m startup_success_predictor.cli export-onnx \
    --checkpoint models/classifier/best.ckpt \
    --input-dim <количество_признаков>

# Запуск инференса через CLI
python -m startup_success_predictor.cli infer \
    --checkpoint models/classifier/best.ckpt \
    --input-csv data/test_sample.csv \
    --output-csv predictions.csv
```

## Развертывание API

### Локально

```bash
uvicorn startup_success_predictor.app:app --reload
# API доступно по адресу http://localhost:8000
# Документация по адресу http://localhost:8000/docs
```

### Docker

```bash
# Сборка и запуск
docker-compose up -d

# Проверка статуса
docker-compose ps

# Просмотр логов
docker-compose logs -f api

# Остановка
docker-compose down
```

## Разработка

### Качество кода

```bash
# Линтинг
ruff check startup_success_predictor/

# Форматирование
ruff format startup_success_predictor/

# Проверка типов
mypy startup_success_predictor/

# Запуск всех проверок
pre-commit run --all-files
```

### Тестирование

```bash
pytest  # Когда тесты будут добавлены
```

## Структура проекта

```
startup_success_predictor/
├── data/
│   ├── download.py      # Загрузка с Kaggle
│   ├── datamodule.py    # PyTorch Lightning DataModule
│   └── preprocessing.py # Предобработка с Polars
├── models/
│   ├── gan_module.py        # WGAN-GP
│   ├── classifier_module.py # MLP Классификатор
│   └── components/          # Архитектуры сетей
├── train.py             # Пайплайн обучения
├── export_onnx.py       # Экспорт модели
├── infer.py             # Скрипт инференса
└── app.py               # Сервис FastAPI
```

## Конфигурация

Редактировать файлы в `configs/` для изменения параметров:

- `data/startup.yaml` - Параметры данных
- `model/gan.yaml` - Архитектура GAN
- `model/classifier.yaml` - Архитектура классификатора
- `train/default.yaml` - Настройки обучения

## Устранение неполадок

### Ошибки импорта

```bash
# Переустановить в режиме редактирования (editable mode)
uv pip install -e .
```

### MLFlow не подключается

```bash
# Проверить, запущен ли сервер
curl http://127.0.0.1:8080/health

# Перезапустить сервер
mlflow server --host 127.0.0.1 --port 8080
```

### Ошибки DVC

```bash
# Реинициализировать DVC
dvc init --force
```

## Следующие шаги

1.  ✅ Настройка проекта завершена
2.  ⏳ Загрузка набора данных
3.  ⏳ Обучение моделей
4.  ⏳ Оценка производительности
5.  ⏳ Развертывание API
6.  ⏳ Мониторинг с MLFlow

## Ресурсы

- [README.md](README.md) - Полная документация
- [IMPLEMENTATION.md](IMPLEMENTATION.md) - Технические детали
- [PDF с заданиями](task/) - Требования к проекту
