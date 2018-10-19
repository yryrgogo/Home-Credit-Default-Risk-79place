import numpy as np
import pandas as pd
import sys
import re
from glob import glob
import os
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
import utils
from utils import logger_func
logger = logger_func()
sys.path.append(f"{HOME}/kaggle/data_analysis/model/")
from params_lgbm import params_home_credit

pd.set_option('max_columns', 200)
pd.set_option('max_rows', 200)


def main():
    submit = sys.argv[1]
    file_path = submit
    utils.submit(file_path=file_path)


if __name__ == '__main__':
    main()
