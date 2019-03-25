import gc
import numpy as np
import pandas as pd
import sys
import re
import glob
import os
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
#  sys.path.append(f"/mnt/c/Git/go/kaggle/github/library/")
import utils
from utils import logger_func
try:
    if not logger:
        logger=logger_func()
except NameError:
    logger=logger_func()

from tqdm import tqdm
import joblib

#========================================================================
# Global Variable
sys.path.append(f"../py")
from info_home_credit import hcdr_key_cols
key, target, ignore_list = hcdr_key_cols()
#========================================================================

#========================================================================
# BigQuery
from google.cloud import bigquery
key_file = f'{HOME}/privacy/horikoshi-ml-a13ef6f2f937.json'
bq_client = bigquery.Client.from_service_account_json(key_file)
#========================================================================

def bq_load(query):
    bq_client = bigquery.Client.from_service_account_json(key_file)
    df = bq_client.query(query).to_dataframe()
    return df


def bq_parallel(query, prefix):
    bq_client = bigquery.Client.from_service_account_json(key_file)
    df = bq_client.query(query).to_dataframe()
    print(f"Result Shape: {df.shape}")

    # prefix = 'f008_big_ins-cur-l3-'
    # prefix = 'f008_big_ins-pre-f3-'
    #  prefix = 'f008_big_ins-pre-l3-'
    base = utils.read_df_pkl('../input/base0*').set_index(key)
    df.set_index(key, inplace=True)
    df = base.join(df)
    utils.save_feature(df_feat=df, ignore_list=ignore_list, is_train=2, prefix=prefix, target=target)


queries = {}
query_file_list = glob.glob("../sql/*.sql")
for query_file in query_file_list:
    with open(query_file) as f:
        query = f.read()

    prefix = utils.get_filename(query_file, delimiter='sql')
    queries[prefix] = query

r = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(bq_parallel)(query, prefix)
    for prefix, query in queries.items()
)
