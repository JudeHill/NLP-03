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

HMM models employ log scale parameters to avoid underflow.

## Getting started

To train and test HMM with EM for 10 epochs and validate every 5 epochs on UPOS tags:

```python
python -m main train-test hmm-EM upos --max-epochs 2 5 --save-path ./save/path.pt --res-path ./res/path.csv
```

Use `--subset` argument to specify the maximum rows of data to be used.

To check more argument usage, run `python -m main --help`.

## Repository structure

```
.
│  argparser.py                 # argument parser
│  hmm_pipeline.py              # HMM training and testing pipelines
│  logging_nlp.py               # logger setup
│  main.py                      # main
│  preprocess_dataset.py        # dataset loading and preprocessing
│  ptb-train.conllu             # Penn Treebank subset dataset
│  python-requirement.txt       # python package requirements
│  README.md
│  utils.py                     # auxiliary functions
│
└─pos_tagging
   │  base.py
   │  hmm.py                    # HMM model
   └─__init__.py
```
