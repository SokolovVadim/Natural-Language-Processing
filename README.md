# Natural-Language-Processing

# P6 - The Apprentice Model

This repository contains the a Natural Language Processing exam project on knowledge distillation. The goal is to train a smaller student model to imitate a larger teacher model on a binary toxic comment classification task. 

Dataset: [`SetFit/toxic_conversations`](https://huggingface.co/datasets/SetFit/toxic_conversations)

Task: binary toxic comment classification.


## Installation

Create and activate a virtual environment:

```
python -m venv .venv
source .venv/bin/activate
```

Install dependencies

pip install -r requirements.txt

## Inspect the Dataset

From the project root, run:

```bash
python scripts/inspect_dataset.py
```

This prints the available dataset splits, column names, sample examples, and label distributions when labels are present.
