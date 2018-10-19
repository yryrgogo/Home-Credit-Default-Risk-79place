import gc
import numpy as np
import pandas as pd
import sys
import re
from glob import glob
import os
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
#  sys.path.append(f"/mnt/c/Git/go/kaggle/github/library/")
import utils
from utils import logger_func, mkdir_func
logger = logger_func()
import eda
from utils import get_categorical_features, get_numeric_features, pararell_process
from feature_engineering import base_aggregation, diff_feature, division_feature, product_feature, cnt_encoding, select_category_value_agg, exclude_feature, target_encoding
from tqdm import tqdm

#========================================================================
# Global Variable
from info_home_credit import hcdr_key_cols
key, target, ignore_list = hcdr_key_cols()
#========================================================================

def interact_feature(df, filekey):
    amt_list = [col for col in df.columns if col.count('revo')]
    day_list = [col for col in df.columns if col.count('DAYS')]

    def combi_caliculate(df, num_list):
        used_list = []
        for f1 in num_list:
            for f2 in num_list:
                if f1==f2:continue
                if sorted([f1, f2]) in used_list:continue
                used_list.append(sorted([f1, f2]))
                df = division_feature(df=df, first=f1, second=f2)
                df = diff_feature(df=df, first=f1, second=f2)
        return df

    #  df = combi_caliculate(df, amt_list)
    #  df = combi_caliculate(df, day_list)

    if filekey.count('bur'):
        df['AMT_CREDIT_SUM_div_AMT_ANNUITY'] = df['AMT_CREDIT_SUM'] / df['AMT_ANNUITY']
        df['AMT_CREDIT_SUM_diff_AMT_CREDIT_DEBT'] = df['AMT_CREDIT_SUM'] - df['AMT_CREDIT_SUM_DEBT']
        df['AMT_CREDIT_SUM_diff_AMT_CREDIT_DEBT_div_AMT_ANNUITY'] = df['AMT_CREDIT_SUM_diff_AMT_CREDIT_DEBT'] / df['AMT_ANNUITY']
        df['AMT_CREDIT_SUM_div_SUM_OVERDUE'] = df['AMT_CREDIT_SUM_OVERDUE'] / df['AMT_CREDIT_SUM']
        df['AMT_CREDIT_SUM_div_MAX_OVERDUE'] = df['AMT_CREDIT_MAX_OVERDUE'] / df['AMT_CREDIT_SUM']

        df['DAYS_CREDIT_ENDDATE_diff_DAYS_CREDIT'] = df['DAYS_CREDIT_ENDDATE'] - df['DAYS_CREDIT']
        df['DAYS_ENDDATE_FACT_diff_DAYS_CREDIT'] = df['DAYS_ENDDATE_FACT'] - df['DAYS_CREDIT']
        df['DAYS_CREDIT_UPDATE_diff_DAYS_CREDIT'] = df['DAYS_CREDIT_UPDATE'] - df['DAYS_CREDIT']

    elif filekey.count('pre'):
        # revoを現在およそ何回支払っているか
        df['revo_AMT_CREDIT_div_AMT_ANNUITY'] = df['revo_AMT_CREDIT'] / df['revo_AMT_ANNUITY']

        # DOWN_PAYMENTに対して他のAMTが占める割合
        df['AMT_CREDIT_div_AMT_DOWN_PAYMENT'] = df['AMT_CREDIT'] / df['AMT_DOWN_PAYMENT']
        df['AMT_ANNUITY_div_AMT_DOWN_PAYMENT'] = df['AMT_ANNUITY'] / df['AMT_DOWN_PAYMENT']

        # revo申請額に対して今どれだけ支払いをしているか
        df['revo_AMT_APPLICATION_div_AMT_CREDIT'] = df['revo_AMT_APPLICATION'] / df['revo_AMT_CREDIT']
        df['revo_AMT_APPLICATION_div_AMT_ANNUITY'] = df['revo_AMT_APPLICATION'] / df['revo_AMT_ANNUITY']
        df['revo_AMT_GOODS_PRICE_div_AMT_CREDIT'] = df['revo_AMT_GOODS_PRICE'] / df['revo_AMT_CREDIT']
        df['revo_AMT_GOODS_PRICE_div_AMT_ANNUITY'] = df['revo_AMT_GOODS_PRICE'] / df['revo_AMT_ANNUITY']

        # ざっくり金利
        df['INTEREST_RATE'] = (df['AMT_CREDIT'] - df['AMT_APPLICATION']) / df['AMT_APPLICATION']
        df['INTEREST_RATE_GOODS'] = (df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']) / df['AMT_GOODS_PRICE']
        df['AMT_INTEREST'] = df['AMT_CREDIT'] - df['AMT_APPLICATION']
        df['AMT_INTEREST_GOODS'] = df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']
        #  df['AMT_CREDIT_div_AMT_ANNUITY'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
        #  df['AMT_CREDIT_div_AMT_ANNUITY_diff_CNT_PAYMENT'] = df['AMT_CREDIT_div_AMT_ANNUITY'] - df['CNT_PAYMENT']

        # DAYS FIRST
        df['DAYS_FIRST_DIFF_DECISION'] = df['DAYS_FIRST_DRAWING'] - df['DAYS_DECISION']
        df['DAYS_FIRST_DIFF_DUE'] = df['DAYS_FIRST_DRAWING'] - df['DAYS_FIRST_DUE']
        df['DAYS_FIRST_DUE_diff_DECISOIN'] = df['DAYS_FIRST_DUE'] - df['DAYS_DECISION']

        # DAYS TERM 
        df['DAYS_LAST_diff_FIRST'] = df['DAYS_LAST_DUE'] - df['DAYS_FIRST_DUE']
        df['DAYS_LAST_1st_diff_FIRST'] = df['DAYS_LAST_DUE_1ST_VERSION'] - df['DAYS_FIRST_DUE']
        df['DAYS_TERMINATION_diff_DECISION'] = df['DAYS_TERMINATION'] - df['DAYS_DECISION']
        df['DAYS_LAST_diff_DECISION'] = df['DAYS_LAST_DUE'] - df['DAYS_DECISION']
        df['DAYS_LAST_1st_diff_DECISION'] = df['DAYS_LAST_DUE_1ST_VERSION'] - df['DAYS_DECISION']

        # DAYS FIRST
        df['revo_DAYS_FIRST_DIFF_DECISION'] = df['revo_DAYS_FIRST_DRAWING'] - df['revo_DAYS_DECISION']
        df['revo_DAYS_FIRST_DIFF_DUE'] = df['revo_DAYS_FIRST_DRAWING'] - df['revo_DAYS_FIRST_DUE']
        df['revo_DAYS_FIRST_DUE_diff_DECISOIN'] = df['revo_DAYS_FIRST_DUE'] - df['revo_DAYS_DECISION']

        # revo_DAYS TERM 
        df['revo_DAYS_LAST_diff_FIRST'] = df['revo_DAYS_LAST_DUE'] - df['revo_DAYS_FIRST_DUE']
        df['revo_DAYS_LAST_1st_diff_FIRST'] = df['revo_DAYS_LAST_DUE_1ST_VERSION'] - df['revo_DAYS_FIRST_DUE']
        df['revo_DAYS_TERMINATION_diff_DECISION'] = df['revo_DAYS_TERMINATION'] - df['revo_DAYS_DECISION']
        df['revo_DAYS_LAST_diff_DECISION'] = df['revo_DAYS_LAST_DUE'] - df['revo_DAYS_DECISION']
        df['revo_DAYS_LAST_1st_diff_DECISION'] = df['revo_DAYS_LAST_DUE_1ST_VERSION'] - df['revo_DAYS_DECISION']

    return df


def make_num_features(df, filekey):
    mkdir_func(f'../features/{filekey}')

    #  if filekey.count('bur'):
    df = interact_feature(df, filekey)

    #========================================================================
    # カテゴリの内容別にNumeric Featureを切り出す
    #========================================================================
    num_list = get_numeric_features(df=df, ignore_list=ignore_list)
    cat_list = get_categorical_features(df=df, ignore_list=[])

    #  few_list = []
    #  for cat in tqdm(cat_list):
    #      for val in tqdm(df[cat].drop_duplicates()):
    #          length = len(df[df[cat]==val])
    #          if length < len(df)*0.002:
    #              few_list.append(val)
    #              continue
    #          for num in num_list:
    #          #  pararell_process(, num_list)
    #              df[f'{num}_{cat}-{val}@'] = df[num].where(df[cat]==val, np.nan)
    #              df[f'{num}_{cat}-fewlist@'] = df[num].where(df[cat].isin(few_list), np.nan)

    logger.info(f'{fname} SET SHAPE : {df.shape}')

    #========================================================================
    # Feature Save & Categorical Encoding & Feature Save 
    #========================================================================
    train = df[~df[target].isnull()]
    test = df[df[target].isnull()]

    categorical_features = get_categorical_features(df=train, ignore_list=ignore_list)

    #========================================================================
    # Numeric Feature Save
    #========================================================================
    for col in train.columns:
        if col in categorical_features:continue
        result_train = train[col].values
        result_test = test[col].values
        logger.info(f"COL: {col} | LENGTH: {len(result_train)}")
        utils.to_pkl_gzip(obj=train[col].values, path=f'../features/{filekey}/train_{col}')
        if col != target:
            utils.to_pkl_gzip(obj=test[col].values, path=f'../features/{filekey}/test_{col}')


def make_cat_features(df, filekey):
    mkdir_func(f'../features/{filekey}')
    train = df[~df[target].isnull()]
    test = df[df[target].isnull()]
    categorical_features = get_categorical_features(df=train, ignore_list=ignore_list)

    #========================================================================
    # Categorical Feature Encode
    #========================================================================
    # Factorize
    logger.info("Factorize Start!!")
    for col in categorical_features:
        for col in categorical_features:
            train[f"lbl_{col}@"], indexer = pd.factorize(train[col])
            test[f"lbl_{col}@"] = indexer.get_indexer(test[col])

    # Count Encoding
    logger.info("Count Encoding Start!!")
    for col in categorical_features:
        train = cnt_encoding(train, col, ignore_list=ignore_list)
        test = cnt_encoding(test, col, ignore_list=ignore_list)

    #========================================================================
    # Categorical Feature Save
    #========================================================================
    for col in train.columns:
        logger.info("Saving Features...")
        if col.count('@'):
            result_train = train[col].values
            result_test = test[col].values
            logger.info(f"COL: {col} | LENGTH: {len(result_train)}")
            utils.to_pkl_gzip(obj=result_train, path=f'../features/{filekey}/train_{col}')
            utils.to_pkl_gzip(obj=result_test, path=f'../features/{filekey}/test_{col}')

'''
#========================================================================
# BUREAU
CREDIT_ACTIVE
Closed      1079273
Active       630607
Sold           6527
Bad debt         21
Name: CREDIT_ACTIVE, dtype: int64
CREDIT_CURRENCY
currency 1    1715020
currency 2       1224
currency 3        174
currency 4         10
Name: CREDIT_CURRENCY, dtype: int64
CREDIT_TYPE
Consumer credit                                 1251615
Credit card                                      402195
Car loan                                          27690
Mortgage                                          18391
Microloan                                         12413
Loan for business development                      1975
Another type of loan                               1017
Unknown type of loan                                555
Loan for working capital replenishment              469
Cash loan (non-earmarked)                            56
Real estate loan                                     27
Loan for the purchase of equipment                   19
Loan for purchase of shares (margin lending)          4
Mobile operator loan                                  1
Interbank credit                                      1
Name: CREDIT_TYPE, dtype: int64
#======================================================================== '''
'''
#========================================================================
# PREVIOUS
Cash loans         747553
Consumer loans     729151
Revolving loans    193164
XNA                   346
Name: NAME_CONTRACT_TYPE, dtype: int64
TUESDAY      223438
WEDNESDAY    223097
MONDAY       222834
FRIDAY       221579
THURSDAY     218463
SATURDAY     216460
SUNDAY       151179
Name: WEEKDAY_APPR_PROCESS_START, dtype: int64
Y    1477046
N          4
Name: FLAG_LAST_APPL_PER_CONTRACT, dtype: int64
XAP                                 729497
XNA                                 677918
Repairs                              23765
Other                                15608
Urgent needs                          8412
Buying a used car                     2888
Building a house or an annex          2693
Everyday expenses                     2416
Medicine                              2174
Payments on other loans               1931
Education                             1573
Journey                               1239
Purchase of electronic equipment      1061
Buying a new car                      1012
Wedding / gift / holiday               962
Buying a home                          865
Car repairs                            797
Furniture                              749
Buying a holiday home / land           533
Business development                   426
Gasification / water supply            300
Buying a garage                        136
Hobby                                   55
Money for a third person                25
Refusal to name the goal                15
Name: NAME_CASH_LOAN_PURPOSE, dtype: int64
Approved        1036781
Canceled         316319
Refused          290678
Unused offer      26436
Name: NAME_CONTRACT_STATUS, dtype: int64
Cash through the bank                        1033552
XNA                                           434220
Non-cash from your account                      8193
Cashless from the account of the employer       1085
Name: NAME_PAYMENT_TYPE, dtype: int64
XAP       1209467
HC         142778
LIMIT       50479
SCO         34538
CLIENT      26431
SCOFR        6109
XNA          3836
VERIF        3329
SYSTEM         83
Name: CODE_REJECT_REASON, dtype: int64
XNA                876255
Unaccompanied      466732
Family             203549
Spouse, partner     64734
Children            30819
Other_B             17116
Other_A              8836
Group of people      2173
Name: NAME_TYPE_SUITE, dtype: int64
Repeater     1064852
New           290678
Refreshed     119795
XNA             1725
Name: NAME_CLIENT_TYPE, dtype: int64
XNA                         758131
Mobile                      224686
Consumer Electronics        121237
Computers                   105748
Audio/Video                  99357
Furniture                    53640
Photo / Cinema Equipment     25018
Construction Materials       24995
Clothing and Accessories     23554
Auto Accessories              7381
Jewelry                       6290
Homewares                     5023
Medical Supplies              3843
Vehicles                      3370
Sport and Leisure             2981
Gardening                     2668
Other                         2553
Office Appliances             2333
Tourism                       1659
Medicine                      1550
Direct Sales                   446
Fitness                        209
Additional Service             128
Education                      107
Weapon                          77
Insurance                       64
House Construction               1
Animals                          1
Name: NAME_GOODS_CATEGORY, dtype: int64
POS     691011
Cash    461563
XNA     324051
Cars       425
Name: NAME_PORTFOLIO, dtype: int64
XNA        1015487
x-sell      361984
walk-in      99579
Name: NAME_PRODUCT_TYPE, dtype: int64
Credit and cash offices       594158
Country-wide                  467925
Stone                         206268
Regional / Local              102602
Contact center                 59987
AP+ (Cash loan)                40904
Channel of corporate sales      4756
Car dealer                       450
Name: CHANNEL_TYPE, dtype: int64
XNA                     701189
Consumer electronics    374400
Connectivity            263933
Furniture                56385
Construction             29441
Clothing                 23462
Industry                 19086
Auto technology           4854
Jewelry                   2668
MLM partners              1121
Tourism                    511
Name: NAME_SELLER_INDUSTRY, dtype: int64
middle        385532
high          353331
XNA           324051
low_normal    322095
low_action     92041
Name: NAME_YIELD_GROUP, dtype: int64
Cash                              285990
POS household with interest       263622
POS mobile with interest          220670
XNA                               193510
Cash X-Sell: middle               143883
Cash X-Sell: low                  130248
POS industry with interest         98833
POS household without interest     82908
Cash Street: high                  59639
Cash X-Sell: high                  59301
Cash Street: middle                34658
Cash Street: low                   33834
POS mobile without interest        24082
POS other with interest            23879
POS industry without interest      12602
POS others without interest         2555
Name: PRODUCT_COMBINATION, dtype: int64
WEDNESDAY    31913
TUESDAY      31680
MONDAY       30723
THURSDAY     30636
FRIDAY       30469
SATURDAY     24171
SUNDAY       13572
Name: revo_WEEKDAY_APPR_PROCESS_START, dtype: int64
Y    184693
N      8471
Name: revo_FLAG_LAST_APPL_PER_CONTRACT, dtype: int64
XAP    193164
Name: revo_NAME_CASH_LOAN_PURPOSE, dtype: int64
XNA    193164
Name: revo_NAME_PAYMENT_TYPE, dtype: int64
XAP       143626
HC         32453
SCOFR       6702
LIMIT       5201
SCO         2929
XNA         1408
SYSTEM       634
VERIF        206
CLIENT         5
Name: revo_CODE_REJECT_REASON, dtype: int64
Unaccompanied      42238
Family              9714
Spouse, partner     2335
Children             747
Other_B              508
Other_A              241
Group of people       67
Name: revo_NAME_TYPE_SUITE, dtype: int64
Repeater     166409
Refreshed     15854
New           10685
XNA             216
Name: revo_NAME_CLIENT_TYPE, dtype: int64
XNA                         192678
Consumer Electronics           339
Audio/Video                     84
Mobile                          22
Computers                       21
Furniture                       16
Photo / Cinema Equipment         3
Other                            1
Name: revo_NAME_GOODS_CATEGORY, dtype: int64
Cards    144985
XNA       48179
Name: revo_NAME_PORTFOLIO, dtype: int64
x-sell     94303
walk-in    50682
XNA        48179
Name: revo_NAME_PRODUCT_TYPE, dtype: int64
Credit and cash offices       125810
Country-wide                   26765
AP+ (Cash loan)                16142
Contact center                 11310
Regional / Local                5926
Stone                           5815
Channel of corporate sales      1394
Car dealer                         2
Name: revo_CHANNEL_TYPE, dtype: int64
XNA                     154531
Consumer electronics     23865
Connectivity             12096
Furniture                 1464
Clothing                   487
Construction               340
Auto technology            136
Industry                   108
MLM partners                94
Jewelry                     41
Tourism                      2
Name: revo_NAME_SELLER_INDUSTRY, dtype: int64
XNA    193164
Name: revo_NAME_YIELD_GROUP, dtype: int64
Card Street    112582
Card X-Sell     80582
Name: revo_PRODUCT_COMBINATION, dtype: int64
#========================================================================
'''

#========================================================================
# Start
#========================================================================
utils.start(sys.argv[0])

app = utils.read_df_pkl(path='../input/clean_app*.p')[[key, target]]

fname_list = [
    #  'bureau'
    'prev'
    #  ,'install'
    #  ,'pos'
    #  ,'ccb'
]
for fname in fname_list:
    logger.info(f"{fname} Start!")
    df_feat = utils.read_df_pkl(path=f'../input/clean_{fname}*.p')

    # Data Check
    #  cat_list = get_categorical_features(df=df_feat, ignore_list=ignore_list)
    #  for col in cat_list:
    #      print(df_feat[col].value_counts())
    #  sys.exit()

    # Target Join
    df_feat = df_feat.merge(app, on=key, how='inner')
    #  df_feat = df_feat.head(10000)

    make_num_features(df=df_feat, filekey=fname)
    make_cat_features(df=df_feat, filekey=fname)

#  pre = utils.read_df_pkl(path='../input/clean_prev*.p')
#  pos = utils.read_df_pkl(path='../input/clean_pos*.p')
#  ins = utils.read_df_pkl(path='../input/clean_ins*.p')
#  ccb = utils.read_df_pkl(path='../input/clean_ccb*.p')

utils.end(sys.argv[0])

#  pre_eda = eda.df_info(pre)
