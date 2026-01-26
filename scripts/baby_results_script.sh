# This is a scratch script intended to mirror the real one and check it works

echo "Starting results computation"
# make directories
# sEM directories (per value for alpha)
mkdir -p ./new_saves/{UPOS,XPOS}/sEM/{alpha_6,alpha_8,alpha_10}
mkdir -p ./new_results/{UPOS,XPOS}/sEM/{alpha_6,alpha_8,alpha_10}
# other directories
mkdir -p ./new_saves/{UPOS,XPOS}/{EM,hardEM,kmeans,NHMM,MLE}
mkdir -p ./new_results/{UPOS,XPOS}/{EM,hardEM,kmeans,NHMM,MLE}
mkdir -p ./figures

# batch EM
python -m main train-test hmm-EM upos --max-epochs 1 1 --save-path ./new_saves/UPOS/EM/10_5.pt --res-path ./new_results/UPOS/EM/10_5.csv
python -m main train-test hmm-EM xpos --max-epochs 1 1 --save-path ./new_saves/XPOS/EM/10_5.pt --res-path ./new_results/XPOS/EM/10_5.csv

# hard EM
python -m main train-test hmm-hardEM upos --max-epochs 1 1 --save-path ./new_saves/UPOS/hardEM/10_5.pt --res-path ./new_results/UPOS/hardEM/10_5.csv
python -m main train-test hmm-hardEM xpos --max-epochs 1 1 --save-path ./new_saves/XPOS/hardEM/10_5.pt --res-path ./new_results/XPOS/hardEM/10_5.csv

# online EM for alpha in [0.6, 0.8, 1.0] (UPOS)
python -m main train-test hmm-sEM upos --max-epochs 1 1 --save-path ./new_saves/UPOS/sEM/alpha_6/10_5.pt --res-path ./new_results/UPOS/sEM/alpha_6/10_5.csv --alpha=0.6
python -m main train-test hmm-sEM upos --max-epochs 1 1 --save-path ./new_saves/UPOS/sEM/alpha_8/10_5.pt --res-path ./new_results/UPOS/sEM/alpha_8/10_5.csv --alpha=0.8
python -m main train-test hmm-sEM upos --max-epochs 1 1 --save-path ./new_saves/UPOS/sEM/alpha_10/10_5.pt --res-path ./new_results/UPOS/sEM/alpha_10/10_5.csv --alpha=1.0

# online EM for XPOS
python -m main train-test hmm-sEM xpos --max-epochs 1 1 --save-path ./new_saves/XPOS/sEM/alpha_6/10_5.pt --res-path ./new_results/XPOS/sEM/alpha_6/10_5.csv --alpha=0.6
python -m main train-test hmm-sEM xpos --max-epochs 1 1 --save-path ./new_saves/XPOS/sEM/alpha_8/10_5.pt --res-path ./new_results/XPOS/sEM/alpha_8/10_5.csv --alpha=0.8
python -m main train-test hmm-sEM xpos --max-epochs 1 1 --save-path ./new_saves/XPOS/sEM/alpha_10/10_5.pt --res-path ./new_results/XPOS/sEM/alpha_10/10_5.csv --alpha=1.0

# k-means
python -m main train-test kmeans upos --max-epochs 1 1 --save-path ./new_saves/UPOS/kmeans/10_5.pt --res-path ./new_results/UPOS/kmeans/10_5.csv
python -m main train-test kmeans xpos --max-epochs 1 1 --save-path ./new_saves/XPOS/kmeans/10_5.pt --res-path ./new_results/XPOS/kmeans/10_5.csv

# NHMM - 10 epochs for speed
python -m main train-test nhmm upos --max-epochs 1 1 --save-path ./new_saves/UPOS/NHMM/10_1.pt --res-path ./new_results/UPOS/NHMM/10_1.csv
python -m main train-test nhmm xpos --max-epochs 1 1 --save-path ./new_saves/XPOS/NHMM/10_1.pt --res-path ./new_results/XPOS/NHMM/10_1.csv

# MLE
python -m main train-test hmm-mle upos --max-epochs 1 1 --save-path ./new_saves/UPOS/MLE/10_5.pt --res-path ./new_results/UPOS/MLE/10_5.csv
python -m main train-test hmm-mle xpos --max-epochs 1 1 --save-path ./new_saves/XPOS/MLE/10_5.pt --res-path ./new_results/XPOS/MLE/10_5.csv

# online EM for 10 epochs for convergence analysis
python -m main train-test hmm-sEM xpos --max-epochs 1 1 --save-path ./new_saves/XPOS/sEM/alpha_6/10_1.pt --res-path ./new_results/XPOS/sEM/alpha_6/10_1.csv --alpha=0.6

# Generate results and figures
echo "Computing results and figures"
python -m scripts.baby_results_generator

# Inspect results for certain cases
echo "Analysing case studies"
python -m scripts.case_studies > case_studies.txt
echo "Case study output written to case_studies.txt"
echo "Done"
