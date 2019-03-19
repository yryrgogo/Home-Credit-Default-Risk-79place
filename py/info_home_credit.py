import numpy as np
import pandas as pd
import sys
import re
import gc
import glob

import os
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/github/library/")
import utils

key = 'SK_ID_CURR'
target = 'TARGET'
key_list = [key, target]
ignore_list = key_list + ['SK_ID_BUREAU', 'SK_ID_PREV']


def hcdr_key_cols():
    return key, target, ignore_list
