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

So i tried to rerun it with larger, therefore more balanced split

Prepared split sizes:
  train: 5000 rows

train label distribution:
  label=0: 3500 (70.00%)
  label=1: 1500 (30.00%)
  validation: 1000 rows

validation label distribution:
  label=0: 700 (70.00%)
  label=1: 300 (30.00%)
  test: 1000 rows

test label distribution:
  label=0: 700 (70.00%)
  label=1: 300 (30.00%)

and got

Training TF-IDF + Logistic Regression baseline...
  train rows: 5000
  validation rows: 1000
  test rows: 1000

Validation metrics:
  accuracy:  0.7610
  precision: 0.5916
  recall:    0.6567
  f1:        0.6224
  confusion_matrix: [[564, 136], [103, 197]]

Test metrics:
  accuracy:  0.7780
  precision: 0.6327
  recall:    0.6200
  f1:        0.6263
  confusion_matrix: [[592, 108], [114, 186]]

ALso decided to save the model in joblib format


## 3. Train small BERT model

installed prajjwal1/bert-tiny

Validation metrics:
  loss:      0.4358
  accuracy:  0.7920
  precision: 0.6949
  recall:    0.5467
  f1:        0.6119
  confusion_matrix: [[628, 72], [136, 164]]

Test metrics:
  loss:      0.4184
  accuracy:  0.8120
  precision: 0.7240
  recall:    0.6033
  f1:        0.6582
  confusion_matrix: [[631, 69], [119, 181]]

Saved metrics to /home/vadim/Github/Natural-Language-Processing/apprentice_model/results/student_baseline_metrics.json
Saved test predictions to /home/vadim/Github/Natural-Language-Processing/apprentice_model/results/student_baseline_predictions.csv
Saved student baseline model to /home/vadim/Github/Natural-Language-Processing/apprentice_model/results/student_baseline

The accuracy is better than TF-IDF but 

Recall is slightly lower

<pre class="overflow-visible! px-0!" data-start="1406" data-end="1432"><div class="relative w-full mt-4 mb-1"><div class=""><div class="contents"><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="relative"><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼs ͼ16"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span>recall: -0.017</span></code></pre></div></div></div></div></div></div></div></div></div></div></div></div></div></pre>
