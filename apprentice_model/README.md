# P6 - The Apprentice Model

NLP exam project on knowledge distillation for toxic comment classification.

The experiment: a supervised `bert-base-uncased` teacher is distilled into a compact `prajjwal1/bert-tiny` student using soft-label temperature distillation (`T=2.0`, `alpha=0.7`) on natural distribution stratified splits from `SetFit/toxic_conversations`.

## Setup

From the repository root, initialize the env

```bash
cd apprentice_model
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Inspect the Hugging Face dataset:

```bash
python scripts/inspect_dataset.py
```

Create the main natural distribution splits:

```bash
python scripts/prepare_natural_splits.py
```

Outputs:

- `data/processed_natural/train.csv`
- `data/processed_natural/validation.csv`
- `data/processed_natural/test.csv`
- `results/natural_split_summary.json`

The older `data/processed/` 70/30 splits are kept only for diagnostics.

## Main Experiments

Train TF-IDF + Logistic Regression on the natural split:

```bash
python scripts/train_tfidf_natural.py
```

Train supervised BERT-tiny:

```bash
python scripts/train_bert_tiny_supervised_natural.py
```

Train supervised BERT-base teacher and export teacher logits:

```bash
python scripts/train_bert_base_supervised.py
```

If the BERT-base model is already trained and you only need to refresh threshold tuning or exported predictions and ogits:

```bash
python scripts/tune_bert_base_threshold.py
```

Validate teacher logits before distillation:

```bash
python scripts/validate_teacher_logits.py
python scripts/smoke_test_distillation_setup.py
```

Train softlabel distilled BERT-tiny:

```bash
python scripts/train_bert_tiny_distilled.py
```

Important outputs:

- `results/tfidf_natural_model.joblib`
- `results/bert_tiny_supervised_natural/`
- `results/bert_base_supervised/`
- `data/teacher_logits/bert_base_train_logits.csv`
- `data/teacher_logits/bert_base_validation_logits.csv`
- `data/teacher_logits/bert_base_test_logits.csv`
- `results/bert_tiny_distilled_t2_a07_natural/`
- `results/bert_tiny_distilled_t2_a07_natural_metrics.json`

## Summaries and Analysis

Create comparison summaries:

```bash
python scripts/create_natural_supervised_comparison.py
python scripts/create_natural_distillation_comparison.py
```

Run the fair CPU benchmark:

```bash
python scripts/benchmark_natural_models_cpu.py --num_repeats 1 --batch_sizes 1 16
```

Create error analysis candidate tables:

```bash
python scripts/create_error_analysis_candidates.py
```

Create token saliency examples:

```bash
python scripts/create_saliency_examples.py
```

Run the small minimal pair bias check:

```bash
python scripts/run_minimal_pair_bias_check.py
```

Generate BERT-base learning curve plots:

```bash
python scripts/plot_training_history.py
```

The main generated analysis files are under `results/`, especially:

- `results/natural_supervised_comparison.*`
- `results/natural_distillation_comparison.*`
- `results/cpu_benchmark_comparison.*`
- `results/error_analysis_candidates.*`
- `results/saliency_examples.*`
- `results/minimal_pair_bias_check.*`
- `results/figures/`
