#!/bin/bash

#python experiment.py
# python experiment.py 3 lgb 0.02 3
<<<<<<< HEAD
# python experiment.py 3 lgb 0.1 150
=======
python make_submission.py 0 xgb 0.02 150 1 3 0
python experiment.py 3 lgb 0.02 605
>>>>>>> 461b84df194f40a7cb879c02f77de533d2815b5d
# python make_submission.py 0 lgb 0.02 150 1 3 0
# python make_submission.py 0 lgb 0.02 150 1 3 1
# python make_submission.py 0 lgb 0.02 250
python simple_keras.py
# gsutil cp '../submit/20180812*' gs://kaggle_port/

# wait
# sudo shutdown -P
