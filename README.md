# Natural Language Processing

Repository for the NLP course project **P6 - The Apprentice Model**.

The main project lives in:

```bash
apprentice_model/
```

Final experiment: distill a supervised `bert-base-uncased` teacher into a compact `prajjwal1/bert-tiny` student for binary toxic comment classification on `SetFit/toxic_conversations`.

## Quick Start

```bash
cd apprentice_model
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Main pipeline:

```bash
python scripts/prepare_natural_splits.py
python scripts/train_tfidf_natural.py
python scripts/train_bert_tiny_supervised_natural.py
python scripts/train_bert_base_supervised.py
python scripts/validate_teacher_logits.py
python scripts/smoke_test_distillation_setup.py
python scripts/train_bert_tiny_distilled.py
python scripts/benchmark_natural_models_cpu.py --num_repeats 1 --batch_sizes 1 16
```

See [`apprentice_model/README.md`](apprentice_model/README.md) for the full script list and outputs.
