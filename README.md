Исследовательская среда для экспериментов с трансформерами на игрушечных языках. Проект позволяет генерировать синтетические датасеты, строить словари, обучать управляемые модели-трансформеры, проводить инференс и анализировать выученные представления.

## Структура репозитория
```
project_root/
├── checkpoints/            # сохранённые веса моделей
├── data/
│   ├── datasets/           # автоматически сгенерированные датасеты (JSONL)
│   ├── dicts/              # словари (txt)
│   └── grammar/            # грамматика и правила
├── logs/                   # логи обучения (TensorBoard/W&B)
├── notebooks/              # примеры исследования в Jupyter
├── reports/                # графики и визуализации
└── src/
    ├── ablations/          # применение структурных абляций
    ├── analysis/           # загрузка модели и аналитические утилиты
    ├── data_generation/    # генератор синтетических корпусов
    ├── inference/          # утилиты для инференса
    ├── models/             # реализация трансформера по блокам
    ├── tokenization/       # токенизаторы и сериализация
    ├── training/           # обучение и управление процессом
    └── vocabulary/         # словарь и его построение
```

## Установка зависимостей
```
pip install -r requirements.txt
```

## Генерация словаря
Постройте JSON-словарь из текстовых файлов со словами (по одному слову на строку):
```
python -m src.vocabulary.cli --inputs data/dicts/*.txt --output data/vocab.json
```
**Параметры:**
- `--inputs` — список путей к текстовым файлам словаря.
- `--output` — путь для сохранения итогового JSON.
- `--lowercase` — привести все токены к нижнему регистру перед добавлением.

Готовый словарь можно загрузить в коде через `Vocabulary.load(path)`.

## Генерация датасетов
Скрипт читает конфигурацию, грамматику и словари, а затем создаёт наборы в `data/datasets/`:
```
python -m src.data_generation.cli --config src/data_generation/config.yaml
```
**Параметры:**
- `--config` — путь к YAML-файлу с настройками генератора (размер датасета, задачи, максимальная длина предложения, сид).
- `--base-dir` — корневая директория проекта (по умолчанию автоматически определяется).

Конфигурация `src/data_generation/config.yaml` задаёт:
- `dataset_size` — количество примеров.
- `max_sentence_length` — максимальная длина предложения.
- `tasks` — список целевых задач (`lm`, `parsing`, `nl2sem`, `sem2nl`).
- `random_seed` — фиксированный сид для воспроизводимости (опционально).

## Токенизация
Модуль `src/tokenization` содержит токенизаторы:
- `WordTokenizer` — разбивка по пробелам, опционально с приведением к нижнему регистру.
- `CharTokenizer` — побуквенная токенизация.
- `BPETokenizer` — простой BPE-токенизатор (потребует предварительного обучения методом `train`).

Для сохранения и загрузки токенизатора используйте функции `save_tokenizer` и `load_tokenizer`:
```python
from src.tokenization import WordTokenizer, save_tokenizer, load_tokenizer
from src.vocabulary import Vocabulary

vocab = Vocabulary.load("data/vocab.json")
tokenizer = WordTokenizer(vocab)
save_tokenizer(tokenizer, "artifacts/tokenizer.json")
restored = load_tokenizer("artifacts/tokenizer.json")
```

## Обучение модели
Запустите обучение через CLI:
```
python -m src.training.cli --config src/config.yaml
```
**Параметры CLI:**
- `--config` — путь к YAML-файлу с полной конфигурацией обучения.

Основные поля `src/config.yaml`:
- `data`
  - `train_path`, `val_path`, `test_path` — пути к JSONL с парами `input`/`output`.
  - `dict_paths` — файлы словаря, используемые для построения `Vocabulary`.
  - `tokenizer` — тип токенизатора (`word` или `char`).
  - `max_seq_len`, `batch_size` — ограничения на длину последовательности и размер батча.
- `model`
  - `architecture` — `encoder_decoder` или `encoder_only`.
  - `d_model`, `n_heads`, `n_layers`, `ffn_dim`, `dropout` — параметры трансформера.
- `training`
  - `num_epochs`, `learning_rate`, `weight_decay`, `gradient_clip` — базовые гиперпараметры.
  - `curriculum` — настройки учебного плана (расписание, стартовая и финальная доли данных).
  - `checkpoint_dir`, `log_dir` — директории для чекпоинтов и логов.
  - `use_wandb`, `wandb_project` — интеграция с Weights & Biases (опционально).

Во время обучения чекпоинты сохраняются в `checkpoint_dir`, а логи TensorBoard пишутся в `log_dir`.

## Инференс
Для получения ответа модели по входной строке используйте гриди-декодер:
```
python -m src.inference.cli checkpoints/model_epoch_019.pt data/vocab.json "пример запроса" --tokenizer word --max-length 64
```
**Аргументы:**
- позиционный `checkpoint` — путь к сохранённой модели.
- позиционный `vocab` — JSON со словарём.
- позиционный `prompt` — входная строка.
- `--tokenizer` — тип токенизатора (`word` или `char`).
- `--max-length` — максимальная длина генерируемой последовательности.

Скрипт автоматически выбирает `cuda`, если доступна, иначе `cpu`, и выводит сгенерированную последовательность в stdout.

## Анализ и абляции
- Модуль `src.analysis.loader` содержит функцию `load_model` для восстановления модели из чекпоинта.
- `src.analysis.attention` — построение тепловых карт внимания и роллаут.
- `src.analysis.embeddings` — извлечение эмбеддингов, понижение размерности (PCA/t-SNE) и визуализация.
- `src.analysis.gradients` — градиентные атрибуции и интегральные градиенты.
- `src.ablations` — применение структурных абляций через `AblationConfig` и функцию `apply_ablation`.

Пример использования в ноутбуке или скрипте Python:
```python
from src.analysis import load_model, extract_embeddings, reduce_embeddings, plot_embeddings
from src.vocabulary import Vocabulary

model, config = load_model("checkpoints/model_epoch_019.pt")
vocab = Vocabulary.load("data/vocab.json")
emb = extract_embeddings(model)
emb2d = reduce_embeddings(emb, method="tsne")
plot_embeddings(emb2d, vocab, save_path="reports/embeddings.png")
```

## Лицензия
MIT
