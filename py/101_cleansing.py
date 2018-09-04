import numpy as np
import pandas as pd
import sys
import re
from glob import glob
import os
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/github/library/")
import utils
from logger import logger_func
logger = logger_func()
import eda

#==============================================================================
# pickleにする 
#==============================================================================

def make_pkl():
    app = pd.read_csv('../input/application_train_test.csv')
    utils.to_df_pickle(df=app, path='../input', fname='application_train_test')

    bureau = pd.read_csv('../input/bureau.csv')
    utils.to_df_pickle(df=bureau, path='../input', fname='bureau')

    prev = pd.read_csv('../input/previous_application.csv')
    utils.to_df_pickle(df=prev, path='../input', fname='previous_application')

    inst = pd.read_csv('../input/installments_payments.csv')
    utils.to_df_pickle(df=inst, path='../input', fname='installments_payments')

    ccb = pd.read_csv('../input/credit_card_balance.csv')
    utils.to_df_pickle(df=ccb, path='../input', fname='credit_card_balance')

    pos = pd.read_csv('../input/POS_CASH_balance.csv')
    utils.to_df_pickle(df=pos, path='../input', fname='POS_CASH_balance')

#  make_pkl()

# DATA LOAD
#  utils.start(sys.argv[0])
#  app = utils.read_df_pickle(path='../input/application*.p').set_index('SK_ID_CURR')
#  bur = utils.read_df_pickle(path='../input/bureau*.p').set_index('SK_ID_CURR')
#  pre = utils.read_df_pickle(path='../input/previous*.p').set_index('SK_ID_CURR')
#  ins = utils.read_df_pickle(path='../input/install*.p').set_index('SK_ID_CURR')
#  ccb = utils.read_df_pickle(path='../input/credit_*.p').set_index('SK_ID_CURR')
#  pos = utils.read_df_pickle(path='../input/POS*.p').set_index('SK_ID_CURR')
#  utils.end(sys.argv[0])

logger.info(f'''
#==============================================================================
# OUTPUT EDA TABLE
#==============================================================================''')
#  app_eda = eda.df_info(app)
#  app_eda.to_csv('../output/application_eda.csv')
#  bur_eda = eda.df_info(bur)
#  bur_eda.to_csv('../output/bureau_eda.csv')
#  pre_eda = eda.df_info(pre)
#  pre_eda.to_csv('../output/prev_eda.csv')
#  ins_eda = eda.df_info(ins)
#  ins_eda.to_csv('../output/install_eda.csv')
#  ccb_eda = eda.df_info(ccb)
#  ccb_eda.to_csv('../output/credit_eda.csv')
#  pos_eda = eda.df_info(pos)
#  pos_eda.to_csv('../output/pos_eda.csv')

utils.start(sys.argv[0])


def clean_app():
    logger.info(f'''
    #==============================================================================
    # APPLICATION CLEANSING
    #==============================================================================''')
    app = utils.read_df_pickle(path='../input/application*.p')
    app['AMT_INCOME_TOTAL'] = app['AMT_INCOME_TOTAL'].where(app['AMT_INCOME_TOTAL']<1000000, 1000000)
    app = utils.to_df_pickle(df=app, path='../input', fname='clean_application_train_test')

def clean_bureau():
    logger.info(f'''
    #==============================================================================
    # BUREAU CLEANSING
    #==============================================================================''')

    bur = utils.read_df_pickle(path='../input/bureau*.p')
    bur['DAYS_CREDIT_ENDDATE'] = bur['DAYS_CREDIT_ENDDATE'].where(bur['DAYS_CREDIT_ENDDATE']>-36000, np.nan)
    bur['DAYS_ENDDATE_FACT'] = bur['DAYS_ENDDATE_FACT'].where(bur['DAYS_ENDDATE_FACT']>-36000, np.nan)
    bur['DAYS_CREDIT_UPDATE'] = bur['DAYS_CREDIT_UPDATE'].where(bur['DAYS_CREDIT_UPDATE']>-36000, np.nan)
    bur = utils.to_df_pickle(df=bur, path='../input', fname='clean_bureau')


def clean_prev():
    logger.info(f'''
    #==============================================================================
    # PREV CLEANSING
    #==============================================================================''')

    cash = 'Cash loans'
    revo = 'Revolving loans'
    ' RevolvingではCNT_PAYMENT, AMT系をNULLにする '
    pre = utils.read_df_pickle(path='../input/previous*.p')
    pre['AMT_CREDIT'] = pre['AMT_CREDIT'].where(pre['AMT_CREDIT']>0, np.nan)
    pre['AMT_ANNUITY'] = pre['AMT_ANNUITY'].where(pre['AMT_ANNUITY']>0, np.nan)
    pre['AMT_APPLICATION'] = pre['AMT_APPLICATION'].where(pre['AMT_APPLICATION']>0, np.nan)
    pre['CNT_PAYMENT'] = pre['CNT_PAYMENT'].where(pre['CNT_PAYMENT']>0, np.nan)
    pre['AMT_DOWN_PAYMENT'] = pre['AMT_DOWN_PAYMENT'].where(pre['AMT_DOWN_PAYMENT']>0, np.nan)
    pre['RATE_DOWN_PAYMENT'] = pre['RATE_DOWN_PAYMENT'].where(pre['RATE_DOWN_PAYMENT']>0, np.nan)

    pre['DAYS_FIRST_DRAWING']        = pre['DAYS_FIRST_DRAWING'].where(pre['DAYS_FIRST_DRAWING'] <100000, np.nan)
    pre['DAYS_FIRST_DUE']            = pre['DAYS_FIRST_DUE'].where(pre['DAYS_FIRST_DUE']         <100000, np.nan)
    pre['DAYS_LAST_DUE_1ST_VERSION'] = pre['DAYS_LAST_DUE_1ST_VERSION'].where(pre['DAYS_LAST_DUE_1ST_VERSION'] <100000, np.nan)
    pre['DAYS_LAST_DUE']             = pre['DAYS_LAST_DUE'].where(pre['DAYS_LAST_DUE']           <100000, np.nan)
    pre['DAYS_TERMINATION']          = pre['DAYS_TERMINATION'].where(pre['DAYS_TERMINATION']     <100000, np.nan)
    pre['SELLERPLACE_AREA']          = pre['SELLERPLACE_AREA'].where(pre['SELLERPLACE_AREA']     <200, 200)

    ignore_list = ['SK_ID_CURR', 'SK_ID_PREV', 'NAME_CONTRACT_TYPE', 'NAME_CONTRACT_STATUS']
    ' revo '
    for col in pre.columns:
        if col in ignore_list:
            logger.info(f'CONTINUE: {col}')
            continue
        pre[f'revo_{col}'] = pre[col].where(pre[f'NAME_CONTRACT_TYPE']==revo, np.nan)

    pre = utils.to_df_pickle(df=pre, path='../input', fname='clean_prev')


pre = utils.read_df_pickle(path='../input/clean_bureau*.p')
print(pre.head())
sys.exit()
pre_eda = eda.df_info(pre)
pre_eda.to_csv('../eda/pre_eda_after.csv')
sys.exit()

ins = utils.read_df_pickle(path='../input/install*.p').set_index('SK_ID_CURR')
ccb = utils.read_df_pickle(path='../input/credit_*.p').set_index('SK_ID_CURR')
pos = utils.read_df_pickle(path='../input/POS*.p').set_index('SK_ID_CURR')

utils.end(sys.argv[0])
