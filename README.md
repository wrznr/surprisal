# Surprisal

A lightweight Python tool for calculating **Surprisal (Information Content)** values for German sentences using GPT-2 language models.

This repository is designed for psycholinguistic research, specifically for generating predictor variables (word-by-word surprisal in bits) for eye-tracking or reading time analyses.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/surprisal.git](https://github.com/your-username/surprisal.git)
    cd surprisal
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The main script is `src/analyzer.py`. It uses `click` for a user-friendly command-line interface.

### 1. Analyze a Single Sentence
Use the `-s` flag to input a sentence directly.

```bash
python src/analyzer.py -s "Wegen ihrer Diät hatte die Gräfin keine Auster nehmen dürfen."
```

**Output:**
```text
Analysis for: Wegen ihrer Diät hatte die Gräfin keine Auster nehmen dürfen.
========================================
  word  surprisal
 Wegen     6.2410
 ihrer     2.1045
  Diät     5.8912
 hatte     3.4501
   die     2.1203
Gräfin    10.5512
 keine     4.8901
Auster    11.2340
nehmen     4.5671
dürfen     1.2301
     .     0.5401
========================================
```

### 2. Process a File (Batch Mode)
You can process a text file (one sentence per line) or a CSV file.

**Text File:**
```bash
python src/analyzer.py -f data/raw/stimuli.txt -o data/processed/results.csv
```

**CSV File:**
If your input is a CSV, specify the column name containing the sentences (default is "sentence").
```bash
python src/analyzer.py -f data/raw/stimuli.csv --column "stimulus_text"
```

## Options

| Flag | Description | Default |
| :--- | :--- | :--- |
| `-s, --sentence` | Input a single sentence string. | *None* |
| `-f, --input-file` | Path to input file (.txt or .csv). | *None* |
| `-o, --output-file`| Path for the output CSV. | `surprisal_output.csv` |
| `-c, --column` | Column name for sentences if input is CSV. | `sentence` |
| `-m, --model` | Hugging Face model name. | `dbmdz/german-gpt2` |
| `--cpu` | Force usage of CPU (even if GPU is available). | `False` |

## Technical Note
The tool uses the **Hugging Face Transformers** library. By default, it uses `dbmdz/german-gpt2`. If a dedicated GPU (NVIDIA or Apple Silicon MPS) is detected, it will automatically use it for faster inference.
