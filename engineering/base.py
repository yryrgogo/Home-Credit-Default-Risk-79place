import numpy as np
import pandas as pd
import datetime
import sys
import re
import glob

sys.path.append('../../../github/module/')
from preprocessing import set_validation, factorize_categoricals
from make_file import make_npy, make_feature_set, make_raw_feature
from convinience_function import get_categorical_features, row_number, get_numeric_features
from logger import logger_func
logger = logger_func()


start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
path_list = glob.glob('../data/*.csv')

unique_id = 'SK_ID_CURR'
target = 'TARGET'
ignore_features = [unique_id, target, 'valid_no', 'is_train', 'is_test']
level_3 = ['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE']


def target_join(path):
    app_tr = pd.read_csv('../data/base_train_test.csv')
    #  app_tr = app_tr[[unique_id, target]]

    data = pd.read_csv(path)
    result = data.merge(app_tr, on=unique_id, how='left')
    print(result.shape)

    result.to_csv(path, index=False)


def make_train_test(train, test, holdout_flg=0):
    #  trn_tes = pd.read_csv('../data/application_train_test.csv')
    ' trainとtestを結合したファイルを作成 '
    train['is_train'] = 1
    train['is_test'] = 0
    test['is_train'] = 0
    test['is_test'] = 1
    test['TARGET'] = -1
    valid = set_validation(train, target, holdout_flg)
    train = train.merge(valid, on=unique_id, how='inner')
    test['valid_no'] = -1
    trn_tes = pd.concat([train, test], axis=0)
    trn_tes.to_csv('../data/application_train_test.csv', index=False)
    sys.exit()


def train_test_valid_no(data, tra_tes):
    '''
    train_testファイルを結合し、is_train/is_test/valid_no/targetを
    結合する
    '''

    data.drop(target, axis=1, inplace=True)
    print(data.shape)
    result = data.merge(tra_tes, on=[unique_id], how='inner')
    print(result.shape)

    result.to_csv('../data/previous_application.csv', index=False)
    col_check(result)


def predicted_distribution():

    train = pd.read_csv('../data/application_train_test.csv')
    path_list = glob.glob('../prediction/*.csv')

    result = pd.DataFrame([])
    for i, path in enumerate(path_list):
        data = pd.read_csv(path)
        data = data.merge(train[[unique_id, target]],
                          on=unique_id, how='inner')

        for col in data.columns:
            if col.count('score'):
                score_name = col
        data.rename(columns={score_name: 'score'}, inplace=True)
        tmp = data[[unique_id, 'prediction', 'score', 'valid_no', target]]
        tmp['score'] = tmp['score'].map(lambda x: f'{str(x)[:6]}_{i}')
        index = np.zeros(len(tmp))
        for i in range(1, int(len(tmp)/40), 1):
            if i == int(len(tmp)/40):
                index[i*40:] = 0
            else:
                index[i*40:(i+1)*40] += i
        print(int(len(tmp)/40))
        tmp['index'] = index
        #  score = str(data[score_name].values[0])[:6]
        #  data.rename(columns = {'prediction':f'{score}'}, inplace=True)

        if len(result) == 0:
            result = tmp
            continue

        result = pd.concat([result, tmp], axis=0)

    result.to_csv(
        f'../output/{start_time[:11]}_predictied_distribution.csv', index=False)

    sys.exit()


def main():

    pd.set_option("display.max_columns", 130)
    pd.set_option("display.max_rows", 130)

    path = '../features/3_winner/*.npy'
    path = '../features/feat_high_cv_overfit/*.npy'
    base = pd.read_csv('../data/base.csv')
    #  base['inf'] = -base[unique_id] / base[target]
    base['inf'] = (base==np.inf).sum()
    print(base.describe().T[['25%', '50%', '75%']])
    sys.exit()
    #  print(base.describe().T)

    logger.info(f'train: {train.shape}')
    logger.info(f'test: {test.shape}')
    train.to_csv('../data/20180827_train_CV08089_LB0809.csv', index=False)
    test.to_csv('../data/20180827_test_CV08089_LB0809.csv', index=False)
    sys.exit()


    #  #  print(base.query('TARGET==-1').iloc[:20, 100:102])
    #  print(base.head())
    #  sys.exit()
    #  data = pd.read_csv('../output/20180826_augument_check.csv')
    #  cv_1 = pd.read_csv('../output/20180825_224_stack_lgb_rate0.02_1099features_CV0.8072030486159842_LB_early150_iter20000.csv')
    #  cv_2 = pd.read_csv('../output/20180825_204_stack_lgb_rate0.02_1099features_CV0.8082070133827914_LB_early150_iter20000.csv')
    #  data['prediction_CV807'] = cv_1[target]
    #  data['prediction_CV808'] = cv_2[target]
    #  data.to_csv('../output/20180826_augument_check.csv', index=False)
    #  sys.exit()

    #  app = pd.read_csv('../data/application_train_test.csv')[[unique_id, 'AMT_CREDIT', 'AMT_ANNUITY', 'EXT_SOURCE_1']]
    #  app['CNT_ANNUITY'] = (app['AMT_CREDIT'].values / app['AMT_ANNUITY'].values)
    #  app['CNT_ANNUITY'] = app['CNT_ANNUITY'].fillna(0).astype('int')
    #  base = pd.read_csv('../data/base.csv')
    #  result=base
    #  path = '../features/3_winner/*.npy'
    #  path = '../features/embedding/*.npy'
    #  #  data = make_feature_set(base[unique_id].to_frame(), path)
    #  data = make_feature_set(base, path)
    #  data['CNT_ANNUITY'] = app['CNT_ANNUITY']
    #  data['EXT_SOURCE_1'] = app['EXT_SOURCE_1']
    #  #  logger.info(data[unique_id].tail(20))
    #  #  logger.info(data.shape)
    #  data.to_csv('../output/20180826_augument_check.csv', index=False)
    #  sys.exit()
    " validation追加 "
    #  print(base.columns)
    #  base[[unique_id, 'valid_no_4']].to_csv('../data/home_credit_Go_validation_index.csv', index=False)
    #  sys.exit()

    #  best_select = pd.read_csv('../output/cv_feature1155_importances_auc_0.807985939108109.csv')
    #  best_feature = best_select['feature'].values
    #  best_feature = [col[5:-10] for col in best_feature if col.count('yuta_')]
    #  use_cols = []
    #  for feature in best_feature:
    #      feature = feature.replace('/', '_').replace(':', '_').replace(' ', '_').replace('.', '_').replace('"', '')
    #      use_cols.append(feature)

    " データセット "
    #  train = pd.read_csv('../data/application_train_test.csv')
    #  train = pd.read_csv('../output/20180822_19_home_credit_train_804features_Go_best_model_CV08028.csv')
    #  train = pd.read_csv('../data/FULL_OLD_BURO_MMM.csv')
    #  train = pd.read_csv('../data/dima_strong_features.csv')
    #  train = pd.read_csv('../data/20180824_07_home_credit_train_833features_Go_best_model_CV08028_regularize_for_NN.csv')
    #  train = pd.read_csv('../data/train_featuretools_1700.csv')
    #  data = pd.read_csv('../data/train_featuretools_1700.csv')
    #  test = pd.read_csv('../data/test_featuretools_1700.csv')
    #  data = pd.read_csv('../features/dima/GPdata_STRATKF_5_SEED605.csv')
    #  test = pd.read_csv('../features/dima/GPtest_STRATKF_5_SEED605.csv')
    #  cols = [ col for col in train.columns if col.count('(')]
    #  train = pd.concat([data, test], axis=0)
    #  train = train[use_cols]
    #  columns = train.columns
    #  re_cols = []
    #  for col in columns:
    #      for use in use_cols:
    #          if col.count(use):
    #              re_cols.append(col)

    #  train = train[re_cols + [unique_id]]
    #  result = base[unique_id].to_frame().merge(train, on=unique_id, how='left')
    #  logger.info(len(result.shape))

    #  for col in cols:
    #      print(col)
    #  sys.exit()
    #  num_list = get_numeric_features(data=train, ignore=ignore_features)
    #  train = train[num_list]
    prefix = 'ccb_'
    prefix = 'a_'
    #  prefix = 'ker_'
    #  prefix = 'dima_'
    #  prefix = 'yuta_'
    #  prefix = 'gp_'
    #  prefix = 'go8028NN_'

    #  check_impute()
    #  for col in train.columns:
    #      if col.count('EXT'):
    #          check_impute(train[[unique_id, col]].set_index(unique_id))
    #  sys.exit()

    ' 各カラムのNULLを確認 '
    #  data = train[ignore_features]
    #  path = '../features/3_winner/*.npy'
    #  base = pd.read_csv('../data/base.csv')
    #  path = '../features/tmptmp/*.npy'
    #  data = make_feature_set(base, path)
    #  #  dataset = dataset.query('is_train==1')
    #  for col in data.columns:
    #      data[col] = data[col].replace(np.inf, np.nan)
    #      data[col].fillna(data[col].mean(), inplace=True)
    #  for col in data.columns:
    #      print(len(data[col][data[col]==np.inf]))
    #      print(len(data[col][data[col].isnull()]))
    #  sys.exit()

    #  predicted_distribution()

    ' fillnaせずにvalidationをセット '
    #  for path in data_list:
    #      data = pd.read_csv(path)
    #      result = set_validation(data, target)
    #      target_join(result, path)
    #  sys.exit()

    #  feature_missing()

    #  result = train
    #  select_list = list(pd.read_csv('../prediction/use_feature/20180531_13_valid2_use_169col_auc_0_7857869389179928.csv')['feature'].values)
    path_list = glob.glob('../output/*.csv')
    path_list = [path for path in path_list if path.count('umap')]
    for path in path_list:

        result = pd.read_csv(path)
        filename = re.search(r'/([^/.]*).csv', path).group(1)
        prefix = f'umap_{filename[9:17]}'
        select_list = []
        make_raw_feature(result, prefix, select_list=select_list, ignore_list=ignore_features,
                         #  path='../features/gp/')
                         #  path='../features/buro_dima/')
                         #  path='../features/3_winner/')
                         #  path='../features/dima/')
                         path='../features/raw_features/')
                         #  path='../features/go/')
    #  #  data = make_feature_set(base[[unique_id, target]], path)
    #  #  data.to_csv('../data/CV08096_dataset_1060features.csv', index=False)
    #  #  base = pd.read_csv('../data/CV08096_dataset_1060features.csv')
    #  #  train = base.query('TARGET>=0')
    #  #  test = base.query('TARGET==-1')
    #  #  test['TARGET'] = np.nan
    #  #  logger.info(train.shape)
    #  #  logger.info(train[target].drop_duplicates())
    #  #  logger.info(test.shape)
    #  #  logger.info(test[target].drop_duplicates())
    #  #  train.to_csv('../data/CV08096_train_1060features.csv', index=False)
    #  #  test.to_csv('../data/CV08096_test_1060features.csv', index=False)

    #  #  sys.exit()


    #  " validation追加 "
    #  #  best_select = pd.read_csv('../output/cv_feature1155_importances_auc_0.807985939108109.csv')
    #  #  best_feature = best_select['feature'].values
    #  #  best_feature = [col[5:-10] for col in best_feature if col.count('yuta_')]
    #  #  use_cols = []
    #  #  for feature in best_feature:
    #  #      feature = feature.replace('/', '_').replace(':', '_').replace(' ', '_').replace('.', '_').replace('"', '')
    #  #      use_cols.append(feature)

    #  " データセット "
    #  #  train = pd.read_csv('../data/application_train_test.csv')
    #  #  train = pd.read_csv('../output/20180822_19_home_credit_train_804features_Go_best_model_CV08028.csv')
    #  #  train = pd.read_csv('../data/FULL_OLD_BURO_MMM.csv')
    #  #  train = pd.read_csv('../data/SPLIT_NEW_PREV_CSV.csv')
    #  #  train = pd.read_csv('../data/SPLIT_NEW_PREV_MMM.csv')
    #  #  train = pd.read_csv('../data/SPLIT_OLD_PREV_CV.csv')
    #  #  train = pd.read_csv('../data/SPLIT_OLD_PREV_SM.csv')
    #  #  train = pd.read_csv('../data/SPLIT_OLD_PREV_MM.csv')
    #  #  train = pd.read_csv('../data/dima_strong_features.csv')
    #  #  train = pd.read_csv('../data/20180824_07_home_credit_train_833features_Go_best_model_CV08028_regularize_for_NN.csv')
    #  #  train = pd.concat([data, test], axis=0)
    #  #  train = train[use_cols]
    #  #  columns = train.columns

    #  #  result = base[unique_id].to_frame().merge(train, on=unique_id, how='left')
    #  #  logger.info(len(result.shape))

    #  #  for col in cols:
    #  #      print(col)
    #  #  sys.exit()
    #  #  num_list = get_numeric_features(data=train, ignore=ignore_features)
    #  #  train = train[num_list]
    #  prefix = 'ccb_'
    #  #  prefix = 'a_'
    #  #  prefix = 'ker_'
    prefix = 'last_dima_'
    #  #  prefix = 'buro_dima_'
    #  #  prefix = 'yuta_'
    #  prefix = 'newb_'
    #  #  prefix = 'gp_'
    #  #  prefix = 'go8028NN_'

    #  #  check_impute()
    #  #  for col in train.columns:
    #  #      if col.count('EXT'):
    #  #          check_impute(train[[unique_id, col]].set_index(unique_id))
    #  #  sys.exit()

    #  ' 各カラムのNULLを確認 '
    #  #  data = train[ignore_features]
    #  #  path = '../features/3_winner/*.npy'
    #  #  base = pd.read_csv('../data/base.csv')
    #  #  path = '../features/tmptmp/*.npy'
    #  #  data = make_feature_set(base, path)
    #  #  #  dataset = dataset.query('is_train==1')
    #  #  for col in data.columns:
    #  #      data[col] = data[col].replace(np.inf, np.nan)
    #  #      data[col].fillna(data[col].mean(), inplace=True)
    #  #  for col in data.columns:
    #  #      print(len(data[col][data[col]==np.inf]))
    #  #      print(len(data[col][data[col].isnull()]))
    #  #  sys.exit()

    #  #  predicted_distribution()

    #  #  result = train
    #  #  select_list = list(pd.read_csv('../prediction/use_feature/20180531_13_valid2_use_169col_auc_0_7857869389179928.csv')['feature'].values)
    #  #  path_list = glob.glob('../output/*.csv')
    #  #  path_list = [path for path in path_list if path.count('umap')]
    #  #  for path in path_list:
    #  result = pd.read_csv('../data/bureau.csv')[[unique_id, 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_ANNUITY']]
    #  result = result.groupby(unique_id).sum()
    #  result_mean = result.groupby(unique_id).mean()
    #  result['CREDIT_per_DEBT'] = result['AMT_CREDIT_SUM'] / result['AMT_CREDIT_SUM_DEBT']
    #  result['CREDIT_per_DEBT_pro_CREDIT_sum'] = result['CREDIT_per_DEBT'] * result['AMT_CREDIT_SUM']
    #  result['CREDIT_per_DEBT_pro_ANNUITY_sum'] = result['CREDIT_per_DEBT'] * result['AMT_ANNUITY']
    #  result['CREDIT_per_DEBT_pro_CREDIT_mean'] = result['CREDIT_per_DEBT'] * result_mean['AMT_CREDIT_SUM']
    #  result['CREDIT_per_DEBT_pro_ANNUITY_mean'] = result['CREDIT_per_DEBT'] * result_mean['AMT_ANNUITY']
    #  drop_list = ['AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_ANNUITY']
    #  result.drop(drop_list, axis=1, inplace=True)
    #  result = base[unique_id].to_frame().merge(result, on=unique_id, how='left')


    data = pd.read_csv('../data/application_train.csv')
    test = pd.read_csv('../data/application_test.csv')
    app = pd.concat([data, test])

    prev = pd.read_csv('../data/previous_application.csv')
    prev['dupl']=prev.duplicated(subset=['NAME_CONTRACT_STATUS', 'NAME_CONTRACT_TYPE', 'AMT_ANNUITY', 'AMT_CREDIT', 'AMT_APPLICATION', 'AMT_GOODS_PRICE', 'AMT_DOWN_PAYMENT', 'RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED', 'NAME_CASH_LOAN_PURPOSE', 'CODE_REJECT_REASON', 'CNT_PAYMENT', 'NAME_YIELD_GROUP', 'NFLAG_INSURED_ON_APPROVAL', 'NAME_PAYMENT_TYPE', 'NAME_CLIENT_TYPE', 'NAME_TYPE_SUITE', 'NAME_PORTFOLIO', 'NAME_GOODS_CATEGORY'],keep=False)

    prev['dupl']=prev['SK_ID_PREV'].where(prev['dupl']==True,np.nan)
    prev['dupl']=prev['dupl'].where(prev['CODE_REJECT_REASON']=='XAP',np.nan)
    prev['dupl']=prev['dupl'].where(prev['AMT_CREDIT']>0,np.nan)

    prev_aggregations = { 'dupl': ['count'], }
    avg_prev = prev.groupby('SK_ID_CURR').agg({**prev_aggregations})
    avg_prev.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in avg_prev.columns.tolist()])
    merged_df=app.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
    result = base[unique_id].to_frame().merge(merged_df, on=unique_id, how='left')

    #  result = pd.read_csv(path)
    #  filename = re.search(r'/([^/.]*).csv', path).group(1)
    #  prefix = f'umap_{filename[9:17]}'
    select_list = []
    make_raw_feature(result, prefix, select_list=select_list, ignore_list=ignore_features,
                     #  path='../features/gp/')
                     #  path='../features/buro_dima/')
                     #  path='../features/3_winner/')
                     #  path='../features/dima/')
                     path='../features/1_third_valid/')
                     #  path='../features/new_dima/')
                     #  path='../features/go/')

    #  make_raw_feature(result, prefix, select_list=[],
    #                   #  path='../features/gp/')
    #                   #  path='../features/buro_dima/')
    #                   #  path='../features/3_winner/')
    #                   path='../features/dima/')
    #                   #  path='../features/go/')


if __name__ == '__main__':
    main()
