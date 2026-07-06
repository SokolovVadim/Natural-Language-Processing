# Progress log


## 1. Load dataset

First got this



 python scripts/prepare_data.py                                ✔  1112  21:05:53
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Repo card metadata block was not found. Setting CardData to empty.
Loaded dataset: SetFit/toxic_conversations
Available splits: ['train', 'test']

- train: 1754874 rows, columns: ['text', 'label', 'label_text']
- test: 50000 rows, columns: ['text', 'label', 'label_text']

Prepared split sizes:
  train: 1000 rows

train label distribution:
  label=0: 902 (90.20%)
  label=1: 98 (9.80%)
  validation: 200 rows

validation label distribution:
  label=0: 183 (91.50%)
  label=1: 17 (8.50%)
  test: 200 rows

test label distribution:
  label=0: 183 (91.50%)
  label=1: 17 (8.50%)

Saved train split to /home/vadim/Github/Natural-Language-Processing/apprentice_model/data/processed/train.csv (1000 rows)
Saved validation split to /home/vadim/Github/Natural-Language-Processing/apprentice_model/data/processed/validation.csv (200 rows)
Saved test split to /home/vadim/Github/Natural-Language-Processing/apprentice_model/data/processed/test.csv (200 rows)


what it looks: 8% is too low for label 1, we need to improve the testing by stratified sampling. And got

Loaded dataset: SetFit/toxic_conversations
Available splits: ['train', 'test']

- train: 1754874 rows, columns: ['text', 'label', 'label_text']
- test: 50000 rows, columns: ['text', 'label', 'label_text']

Prepared split sizes:
  train: 1000 rows

train label distribution:
  label=0: 700 (70.00%)
  label=1: 300 (30.00%)
  validation: 200 rows

validation label distribution:
  label=0: 140 (70.00%)
  label=1: 60 (30.00%)
  test: 200 rows

test label distribution:
  label=0: 140 (70.00%)
  label=1: 60 (30.00%)

### 2. Training baseline

Training TF-IDF + Logistic Regression baseline...
  train rows: 1000
  validation rows: 200
  test rows: 200

Validation metrics:
  accuracy:  0.7350
  precision: 0.5854
  recall:    0.4000
  f1:        0.4752
  confusion_matrix: [[123, 17], [36, 24]]

Test metrics:
  accuracy:  0.6850
  precision: 0.4694
  recall:    0.3833
  f1:        0.4220
  confusion_matrix: [[114, 26], [37, 23]]

Saved metrics to /home/vadim/Github/Natural-Language-Processing/apprentice_model/results/tfidf_baseline_metrics.json
Saved test predictions to /home/vadim/Github/Natural-Language-Processing/apprentice_model/results/tfidf_baseline_predictions.csv

And we can see that it's not strong at all
