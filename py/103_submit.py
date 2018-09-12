import numpy as np
import pandas as pd
import sys
import re
from glob import glob
import os
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/github/library/")
import utils
from utils import logger_func
logger = logger_func()
pd.set_option('max_columns', 200)
pd.set_option('max_rows', 200)



def main():
    file_path = '../submit/20180911_074_submit_lgb_rate0.02_1138features_CV0.8029377226175965_LB_early150_iter20000.csv'
    utils.submit(file_path=file_path)


if __name__ == '__main__':
    main()
