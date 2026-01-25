# Unsupervised Learning for Part-of-Speech Tagging

This project employs a subset of Penn Treebank dataset and evaluates
HMM and K-means on the PoS tagging problem.

## Introduction

This project implements the following algorithms:

- Hidden Markov Model (HMM) + Expectation-Maximization (EM) algorithms:
  - Standard EM (EM) (the classic EM algorithm)
  - Stochastic EM (sEM)
  - Viterbi-EM (hard-EM)
  - Maximal Likelihood Estimation (MLE) (supervised learning)
- K-means clustering
- Neural HMM (Tran et al., 2016)

HMM models employ log scale parameters to avoid underflow.


## Setup (Python Virtual Environment)

### 1. Create a virtual environment
~~~bash
python -m venv venv
~~~

### 2. Activate the virtual environment
- **macOS / Linux**
~~~bash
source venv/bin/activate
~~~

- **Windows (PowerShell)**
~~~powershell
venv\Scripts\Activate.ps1
~~~

### 3. Install dependencies
~~~bash
pip install -r requirements.txt
~~~

### 4. Deactivate (when done)
~~~bash
deactivate
~~~

## Training and testing

To train and test HMM with EM for 10 epochs and validate every 5 epochs on UPOS tags:

```python
python -m main train-test hmm-EM upos --max-epochs 2 5 --save-path ./save/path.pt --res-path ./res/path.csv
```

Use `--subset` argument to specify the maximum rows of data to be used.

To check more argument usage, run `python -m main --help`.

## Repository structure

├── README.md
├── argparser.py
├── figs
├── hmm_pipeline.py      
├── kmeans_pipeline.py
├── logging_nlp.py
├── logs
├── main.py
├── nhmm_pipeline.py
├── pos_tagging
│   ├── __init__.py
│   ├── base.py
│   ├── data_loader.py
│   ├── gpu_check.py
│   ├── hmm.py                 
│   ├── kmeans.py
│   └── neural_hmm.py
├── preprocess_dataset.py
├── ptb-train.conllu
├── python-requirement.txt
├── requirements.txt
├── results_generator_2.ipynb
├── results_generator_old.py
├── scripts
│   ├── baby_results_generator.py
│   ├── baby_results_script.sh
│   ├── case_studies.py
│   ├── results_generator.py
│   └── results_script.sh
├── upos_script.bat
├── utils.py
└── xpos_script.bat

## Reproducing results
To reproduce results, run 

~~~bash
source scripts/results_script.sh
~~~

from the parent directory of the project.

This will re-run all methods of training for UPOS and XPOS tags, for 50 epochs in each case (apart from the NHMM, which is run for 10 epochs). A python script is then run to produce figures (saved to figures/) which show convergence for each method, and a pandas DF which closely mirrors the results table included in the report (this is printed to the console).

Another python script then uses Pandas to analyse the CSV output from each method, accumulating data about the sentences on which models performed best and worst. Selections of this output which mirrors what was included in the final report is then written to case_studies.txt

__NOTE__: Each run of results is for 50 epochs, so running all consecutively will likely take in the order of days, depending on hardware. A VM is strongly recommended to run this project. 