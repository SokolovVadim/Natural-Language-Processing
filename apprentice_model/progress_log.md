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

## 4. Add benchmarks

Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41/41 [00:00<00:00, 8155.09it/s]

Benchmark results:

TF-IDF + Logistic Regression
  size: 0.89 MB
  examples: 1000
  avg total inference time: 0.0563 sec
  avg time/example: 0.0563 ms
  examples/sec: 17758.85
  device: cpu

BERT-tiny Student
  size: 17.42 MB
  examples: 1000
  avg total inference time: 1.6386 sec
  avg time/example: 1.6386 ms
  examples/sec: 610.29
  device: cpu
  batch size: 32

Saved benchmark JSON to /home/vadim/Github/Natural-Language-Processing/apprentice_model/results/benchmark_results.json
Saved benchmark CSV to /home/vadim/Github/Natural-Language-Processing/apprentice_model/results/benchmark_results.csv

SO we can see that TF-IDF is more efifcient than BERT

## 5. Connect openAI teacher

I added the open AI teacher via tocken

And tried to authorize then got a problem with the billing, fixed and ran the training. 

Example:


python scripts/generate_teacher_labels.py --split train --limit 1

Labeling train: target=1
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:07<00:00,  7.26s/it]
/home/vadim/Github/Natural-Language-Processing/apprentice_model/scripts/generate_teacher_labels.py:353: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
  existing_labels = pd.concat(

Summary for train:
  labeled examples: 1
  teacher label distribution:
    label=0: 1
  agreement with original labels: 0.00%
  disagreements: 1
    train:0:1c458b809a26aa60: original=1 teacher=0 text='Damnit!, I was going to Pearson airport Thursday morning to have my bag handled. Well that sucks!!........Hmmmm, maybe.'


Then identified a problem that the teacher labeling currently selects the first N rows instead of a stratified sample

## 6. Train, validate and test teacher

### Train

python scripts/generate_teacher_labels.py --split train --limit 10

Labeling train: target=10
Selected original label distribution:
  label=0: 7
  label=1: 3
 90%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍             | 9/10 [00:59<00:07,  7.13s/it]/home/vadim/Github/Natural-Language-Processing/apprentice_model/scripts/generate_teacher_labels.py:468: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
  combined = pd.concat(
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [01:04<00:00,  6.43s/it]
/home/vadim/Github/Natural-Language-Processing/apprentice_model/scripts/generate_teacher_labels.py:477: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
  existing_labels = pd.concat(

Summary for train:
  labeled examples: 10
  teacher label distribution:
    label=0: 5
    label=1: 5
  agreement with original labels: 80.00%
  disagreements: 2
    train:3183:9152f85808ba4792: original=0 teacher=1 text='You keep talking about "you people," and you talk about Scalia calling for the end of "The Rule of Law" in America. Ther'
    train:4584:a283fef386f61485: original=0 teacher=1 text='Again the globe and mail goes over the top with its vile race baiting hate propaganda.'

### Validate


python scripts/generate_teacher_labels.py --split validation --limit 10

Labeling validation: target=10
Selected original label distribution:
  label=0: 7
  label=1: 3
 90%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍             | 9/10 [01:04<00:07,  7.36s/it]/home/vadim/Github/Natural-Language-Processing/apprentice_model/scripts/generate_teacher_labels.py:468: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
  combined = pd.concat(
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [01:10<00:00,  7.04s/it]
/home/vadim/Github/Natural-Language-Processing/apprentice_model/scripts/generate_teacher_labels.py:477: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
  existing_labels = pd.concat(

Summary for validation:
  labeled examples: 10
  teacher label distribution:
    label=0: 7
    label=1: 3
  agreement with original labels: 80.00%
  disagreements: 2
    validation:152:0cf43bdc4ee9ed7f: original=1 teacher=0 text='Someone in LA is prejudice against Asians? Why? look too much like Mexicans? Actually she should be made to write a pape'
    validation:311:8bc6791edddaec65: original=0 teacher=1 text="Time Bandit you don't have a clue what drives a thriving economy its not REDISTRIBUTION OF WEALTH in this United States."

### Test


python scripts/generate_teacher_labels.py --split test --limit 10

Labeling test: target=10
Selected original label distribution:
  label=0: 7
  label=1: 3
 90%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍             | 9/10 [00:50<00:05,  5.19s/it]/home/vadim/Github/Natural-Language-Processing/apprentice_model/scripts/generate_teacher_labels.py:468: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
  combined = pd.concat(
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:54<00:00,  5.42s/it]
/home/vadim/Github/Natural-Language-Processing/apprentice_model/scripts/generate_teacher_labels.py:477: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
  existing_labels = pd.concat(

Summary for test:
  labeled examples: 10
  teacher label distribution:
    label=0: 3
    label=1: 7
  agreement with original labels: 60.00%
  disagreements: 4
    test:951:ccfc6c5282949ffa: original=0 teacher=1 text='Are you sitting down? In his latest "Love Me" rally, Trump said the following: “With the exception of the late, great Ab'
    test:972:5db1b876bd1b5f6c: original=0 teacher=1 text='Canadians have called for and will support action on climate change. The carbon tax is entirely within federal jurisdict'
    test:311:1f25df3e9046ce2d: original=0 teacher=1 text="More yapping from someone who knows nothing about the big picture. Can't see the forest for the trees can you?"
    test:537:78b8d314a7b6857a: original=0 teacher=1 text='A few years ago, I was told that an autopsy showed that a close personal friend of mine had died of natural causes and w'

This is okay for a tiny sample	


## 7. Check a larger set


python scripts/generate_teacher_labels.py --split train --limit 100

Labeling train: target=100
Selected original label distribution:
  label=0: 70
  label=1: 30
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [12:38<00:00,  7.58s/it]

Summary for train:
  labeled examples: 100
  teacher label distribution:
    label=0: 51
    label=1: 49
  agreement with original labels: 75.00%
  disagreements: 25
    train:3183:9152f85808ba4792: original=0 teacher=1 text='You keep talking about "you people," and you talk about Scalia calling for the end of "The Rule of Law" in America. Ther'
    train:4584:a283fef386f61485: original=0 teacher=1 text='Again the globe and mail goes over the top with its vile race baiting hate propaganda.'
    train:2327:b4612b3729c149f8: original=0 teacher=1 text='Are you serious? You give recognition to whom and for what? If it is meant to be, so be it. It is part of life - grow up'
    train:1470:51f2ddb18611d49b: original=1 teacher=0 text="He woulda got away with it to if it wasn'the for those darn pesky kids and that dog."
    train:2384:a657dc22c14427ef: original=0 teacher=1 text='From the article. "The government must therefore also encourage people to refrain from using legal pot until they are 25'

python scripts/generate_teacher_labels.py --split validation --limit 50

Labeling validation: target=50
Selected original label distribution:
  label=0: 35
  label=1: 15
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [04:36<00:00,  5.52s/it]

Summary for validation:
  labeled examples: 50
  teacher label distribution:
    label=0: 27
    label=1: 23
  agreement with original labels: 80.00%
  disagreements: 10
    validation:152:0cf43bdc4ee9ed7f: original=1 teacher=0 text='Someone in LA is prejudice against Asians? Why? look too much like Mexicans? Actually she should be made to write a pape'
    validation:311:8bc6791edddaec65: original=0 teacher=1 text="Time Bandit you don't have a clue what drives a thriving economy its not REDISTRIBUTION OF WEALTH in this United States."
    validation:847:96b1d08ed35bfd61: original=0 teacher=1 text='I\'m actually enjoying the usual string of "sour grapes!" comments from those who have never managed to get a job in jour'
    validation:695:4bf996aefaa559b3: original=0 teacher=1 text='Congrats alt-right neonazi sympathizer. You scored a "most disagreed with" commenter status.'
    validation:593:2d238397ffd16d62: original=0 teacher=1 text='As is the case with Islam when they call for the death of America, why is it so hard to understand when North Korea asks'

 python scripts/generate_teacher_labels.py --split test --limit 50

Labeling test: target=50
Selected original label distribution:
  label=0: 35
  label=1: 15
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [05:04<00:00,  6.09s/it]

Summary for test:
  labeled examples: 50
  teacher label distribution:
    label=0: 25
    label=1: 25
  agreement with original labels: 80.00%
  disagreements: 10
    test:951:ccfc6c5282949ffa: original=0 teacher=1 text='Are you sitting down? In his latest "Love Me" rally, Trump said the following: “With the exception of the late, great Ab'
    test:972:5db1b876bd1b5f6c: original=0 teacher=1 text='Canadians have called for and will support action on climate change. The carbon tax is entirely within federal jurisdict'
    test:311:1f25df3e9046ce2d: original=0 teacher=1 text="More yapping from someone who knows nothing about the big picture. Can't see the forest for the trees can you?"
    test:537:78b8d314a7b6857a: original=0 teacher=1 text='A few years ago, I was told that an autopsy showed that a close personal friend of mine had died of natural causes and w'
    test:847:afe63df65849a9a9: original=0 teacher=1 text='I think the public deserves to know what the original tax bill was. In other words, how many PFD checks worth in tax lia'

 train
Rows: 100
Original:
original_label
0    70
1    30
Name: count, dtype: int64
Teacher:
teacher_label
0    51
1    49
Name: count, dtype: int64
Agreement: 0.75

 validation
Rows: 50
Original:
original_label
0    35
1    15
Name: count, dtype: int64
Teacher:
teacher_label
0    27
1    23
Name: count, dtype: int64
Agreement: 0.8

 test
Rows: 50
Original:
original_label
0    35
1    15
Name: count, dtype: int64
Teacher:
teacher_label
1    25
0    25
Name: count, dtype: int64
Agreement: 0.8

Original labels are 70/30, but teacher labels are closer to 50/50:

This means the OpenAI teacher often marks comments as toxic even when the dataset says it's not.
