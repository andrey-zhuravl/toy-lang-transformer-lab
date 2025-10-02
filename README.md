# Toy Language Transformer Lab

Исследовательская среда для экспериментов с трансформерами на игрушечных языках. Проект позволяет генерировать синтетические датасеты, обучать управляемые модели-трансформеры, проводить абляции и анализировать выученные представления.

## Структура репозитория
```
project_root/
├── checkpoints/            # сохранённые веса моделей
├── data/
│   ├── datasets/           # автоматически сгенерированные датасеты (JSONL)
│   ├── dicts/              # словари (txt)
│   └── grammar/            # грамматика и правила
├── logs/                   # логи обучения (TensorBoard/W&B)
├── notebooks/
│   ├── train.ipynb         # пример обучения
│   └── analyze.ipynb       # пример анализа
├── reports/                # графики и визуализации
└── src/                    # исходный код
    ├── ablations.py
    ├── analyzer.py
    ├── config.yaml
    ├── generator.py
    ├── model.py
    ├── tokenizer.py
    ├── trainer.py
    └── vocab.py
```

## Быстрый старт
1. Создайте словари и грамматику в каталоге `data/` или используйте примеры.
2. Сгенерируйте датасеты:
   ```bash
   python -m src.generator --task lm --split train --num-samples 1000
   ```
3. Настройте параметры в `src/config.yaml`.
4. Запустите обучение через ноутбук `notebooks/train.ipynb` или напрямую:
   ```bash
   python -m src.trainer --config src/config.yaml
   ```
5. Проанализируйте модель с помощью `notebooks/analyze.ipynb` или скрипта `src/analyzer.py`.

## Зависимости
- Python 3.10+
- PyTorch 2.x
- numpy, tqdm, pyyaml, matplotlib, seaborn, scikit-learn, tensorboard
- (опционально) wandb

Установите зависимости:
```bash
pip install -r requirements.txt
```

## Лицензия
MIT
