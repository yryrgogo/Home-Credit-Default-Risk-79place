import gc
import numpy as np
import pandas as pd
import datetime
from datetime import date, timedelta
import glob
import sys
import re
import shutil
from itertools import combinations
from multiprocessing import Pool
import multiprocessing
from tqdm import tqdm

sys.path.append('../../../github/module/')
from preprocessing import set_validation, split_dataset, factorize_categoricals, get_dummies, data_regulize, max_min_regularize, inf_replace, data_regulize
from load_data import pararell_load_data, x_y_split
from convinience_function import get_categorical_features, get_numeric_features
from logger import logger_func
from make_file import make_feature_set, make_npy
from statistics_info import correlation
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
logger = logger_func()

unique_id = 'SK_ID_CURR'
target = 'TARGET'
ignore_features = [unique_id, target, 'valid_no', 'valid_no_2',
                   'valid_no_3', 'valid_no_4', 'is_train', 'is_test']

pd.set_option('max_columns', 100)


def over_sampling(x_train, y_train, positive_count):

    logger.info(f'{x_train.shape}')
    # SMOTEで不正利用の割合を約10%まで増やす
    smote = SMOTEENN(ratio={0:positive_count*90, 1:positive_count*10}, random_state=0)
    x_train_resampled, y_train_resampled = smote.fit_sample(x_train, y_train)

    logger.info('y_train_resample:\n{}'.format(pd.Series(y_train_resampled).value_counts()))

    for col in x_train_resampled.columns:
        logger.info(col)
    logger.info(y_train_resampled)

    logger.info(x_train_resampled)
    x_train_resampled['over_sample'] = y_train_resampled

    x_train_resampled.to_csv(f'../data/{start_time}over_sampling.csv', index=False)

    sys.exit()


def main():

    path = '../features/3_winner/*.npy'
    base = pd.read_csv('../data/base.csv')
    data = make_feature_set(base[[unique_id, 'TARGET', 'valid_no_4']], path)
    #  app['AMT_CREDIT'] = app['AMT_CREDIT'].where( app['NAME_CONTRACT_TYPE'] == 'Cash loans', np.nan)
    #  app['AMT_ANNUITY'] = app['AMT_ANNUITY'].where( app['NAME_CONTRACT_TYPE'] == 'Cash loans', np.nan)
    #  app['length'] = app['AMT_CREDIT']/app['AMT_ANNUITY']

    positive1 = data.query("23.94<=dima_length<=24.26").query('10000<=dima_AMT_ANNUITY<=36800')[unique_id].values
    positive2 = data.query("23.76<=dima_length<=23.82").query('4700<=dima_AMT_ANNUITY<=18000')[unique_id].values
    positive3 = data.query("26.38<=dima_length<=26.55").query('25190<=dima_AMT_ANNUITY<=43920')[unique_id].values
    positive4 = data.query("28.65<=dima_length<=28.80").query('31500<=dima_AMT_ANNUITY<=45440')[unique_id].values
    positive5 = data.query("28.57<=dima_length<=29.32").query('5800<=dima_AMT_ANNUITY<=24150')[unique_id].values
    positive6 = data.query("32.0<=dima_length<=32.26").query('41500<=dima_AMT_ANNUITY<=67100')[unique_id].values
    positive7 = data.query("17.95<=dima_length<=18.27").query('8500<=dima_AMT_ANNUITY<=102200')[unique_id].values

    positive_id = list(set(list(np.hstack((positive1, positive2, positive3, positive4, positive5, positive6, positive7)))))

    negative1 = data.query("23.0<=dima_length<=26.5").query('4500<=dima_AMT_ANNUITY<=45800')[unique_id].values
    negative2 = data.query("28.55<=dima_length<=29.40").query('5000<=dima_AMT_ANNUITY<=45440')[unique_id].values
    negative3 = data.query("32.0<=dima_length<=32.50").query('41000<=dima_AMT_ANNUITY<=68100')[unique_id].values
    negative4 = data.query("17.50<=dima_length<=18.50").query('8500<=dima_AMT_ANNUITY<=102200')[unique_id].values

    negative_id = list(set(list(np.hstack((negative1, negative2, negative3, negative4)))))

    logger.info(len(positive_id))
    logger.info(len(negative_id))
    #  use_cols = get_numeric_features(data=app, ignore=ignore_features)
    #  app = app[use_cols]
    #  app = app.replace(np.inf, np.nan)
    #  app = app.replace(-1*np.inf, np.nan)
    #  app = app.fillna(0)
    df_p = data.loc[positive_id, :]
    df_n = data.loc[negative_id, :]

    positive_count = len(df_p)
    target = 'over_sample'
    df_p['over_sample'] = 1
    df_n['over_sample'] = 0
    df = pd.concat([df_p, df_n], axis=0)

    df = data_regulize(df=df, na_flg=1, inf_flg=1, mm_flg=1, float16_flg=1, ignore_feature_list=ignore_features, logger=logger)

    #  for col in df.columns:
    #      df[col] = df[col].replace(float(np.inf), np.nan)
    #      df[col] = df[col].replace(float(-1*np.inf), np.nan)
    #      if len(df[col][df[col].isnull()])>0:
    #          df[col] = df[col].fillna(df[col].median(), inplace=True)
    #  df = df.astype('float16')

    x_train, y_train = x_y_split(df, target)

    #  logger.info(x_train.shape)
    #  logger.info(x_train.head())
    #  logger.info(y_train[:20])

    over_sampling(x_train, y_train, positive_count)


if __name__ == '__main__':

    main()
