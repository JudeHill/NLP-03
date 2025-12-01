python -m main train-test hmm-EM upos --max-epochs 9 5 --save-path ./save/EM/hmm_45.pt --res-path ./res/EM/9_5.csv
python -m main train-test hmm-hardEM upos --max-epochs 9 5 --save-path ./save/hardEM/hmm_45.pt --res-path ./res/hardEM/9_5.csv
python -m main train-test hmm-sEM upos --max-epochs 9 5 --save-path ./save/sEM/hmm_45.pt --res-path ./res/sEM/9_5.csv



