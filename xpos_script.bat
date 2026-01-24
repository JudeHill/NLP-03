
python -m main train-test hmm-EM xpos --max-epochs 9 5 --save-path ./xpos_save/EM/hmm_45.pt --res-path ./xpos_res/EM/9_5.csv
python -m main train-test hmm-hardEM xpos --max-epochs 9 5 --save-path ./xpos_save/hardEM/hmm_45.pt --res-path ./xpos_res/hardEM/9_5.csv
python -m main train-test hmm-sEM xpos --max-epochs 9 5 --save-path ./xpos_save/sEM/hmm_45.pt --res-path ./xpos_res/sEM/9_5.csv