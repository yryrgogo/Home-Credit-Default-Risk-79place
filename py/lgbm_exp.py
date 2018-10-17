import gc
import numpy as np
import pandas as pd
import sys
import re
from glob import glob
import os
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/github/library/")
#  sys.path.append(f"/mnt/c/Git/go/kaggle/github/library/")
import utils
from utils import logger_func
logger = logger_func()

key = 'SK_ID_CURR'
target = 'TARGET'

utils.start(sys.argv[0])

app = utils.read_df_pickle(path='../input/application_train_test000.p')[[key, target, 'AMT_ANNUITY', 'EXT_SOURCE_2', 'DAYS_BIRTH']].dropna()
df = app.sample(5000)

print(df)
sys.exit()







# ==================================================
utils.end(sys.argv[0])

