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
import shutil

pd.set_option('max_columns', 200)
pd.set_option('max_rows', 200)


def main():
    submit = sys.argv[1]
    try:
        comment = sys.argv[2]
        utils.submit(file_path=submit, comment=comment)
    except IndexError:
        utils.submit(file_path=submit)

    shutil.move(submit, '../log_submit/')

if __name__ == '__main__':
    main()
